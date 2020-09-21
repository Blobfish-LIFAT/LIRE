"""
    This is an attempt to produce a simple Numpy based version of the whole code
    This program implements
        - a new approach for RS explanation based on LIME. Contrary to LIME-RS
            - interpretable space is no more binary
            - we are closer to LIME philosophy since our pertuvbed entries in interpretable space do not
            directly coincidate with pre-existing points
            - as a consequence we propose a new out-of-sample prediction method to retrieve the prediction for all
            perturbed points as a quadratic error minimization problem
            - we change the definition of locality to better embrace potential decision boundaries
        - a new robustness measure for RS explanation
    What changes:
        - only a single approach implemented: LIRE with varying percentage of training points
    either from cluster or from perturbed points
        - no more torch representation, and no more use of Adam optimizer to determine out-of-sample predictions
"""


## imports
import numpy as np
import os.path
import umap
from utility import load_data
from scipy.optimize import least_squares
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt

## constants
VERBOSE = False
F_LATENT_FACTORS = 10
DIM_UMAP = 3            # number of dimensions after UMAP reduction
N_TRAINING_POINTS = 50
PERT_STD = 1.04 #empiricaly determined from dataset
#MOVIE_IDS = random.sample(range(2000), 10)
#USER_IDS = random.sample(range(610), 10)

def oos_predictor(perturbation, sigma, Vt):
    def prepare(perturbation):
        """
        construct residual computation for least square optimization
        :param perturbation: oos user signature
        :return: method to compute residual vector for oos user "perturbation"
        """
        def pred_fn(user_vec_lat):
            """
            Compute an upgrade to the latent representation as a residual
            :param user_vec_lat: current latent space representation of perturbation oos user
            :return:
            """
            umean = user_vec_lat.sum() / (user_vec_lat != 0.).sum()
            umask = perturbation != 0.
            pred = (user_vec_lat @ sigma @ Vt) + umean
            return (perturbation - pred) * umask

        return pred_fn

    res = least_squares(prepare(perturbation), np.ones(sigma.shape[0])) # latent space dimension

    moy = perturbation.sum() / (perturbation != 0.).sum()

    return (res.x @ sigma @ Vt) + moy


## code
if __name__ == '__main__':
    U = None
    sigma = None
    Vt = None
    all_user_predicted_ratings = None

    # 1. Loading data and setting all matrices
    if os.path.isfile("U.gz") and os.path.isfile("sigma.gz") and os.path.isfile("Vt.gz") \
            and os.path.isfile("all_ratings.gz"):
        # loading pre-computed matrices
        U = np.loadtxt("U.gz")
        sigma = np.loadtxt("sigma.gz")
        Vt = np.loadtxt("Vt.gz")
        all_user_predicted_ratings = np.loadtxt("all_ratings.gz")
        labels = np.loadtxt("labels.gz")

    else:
        # 1. loading and setting data matrices
        U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, \
            movies_df, ratings_df, films_nb = load_data()

        if VERBOSE: print("films", films_nb)

        # saving matrices
        np.savetxt("U.gz", U)
        np.savetxt("sigma.gz", sigma)
        np.savetxt("Vt.gz", Vt)
        np.savetxt("all_ratings.gz", all_user_predicted_ratings)

        ## 2. Determining neighborhood as a clustering problem
        reducer = umap.UMAP(n_components=3, n_neighbors=20, random_state=12)  # metric='cosine'
        embedding = reducer.fit_transform(np.nan_to_num(all_actual_ratings))

        ## 3. Clustering
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(embedding)
        labels = clusterer.labels_
        np.savetxt("labels.gz")

        # visualization of neighborhood
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                   c=[sns.color_palette("RdBu", n_colors=max(clusterer.labels_) - 1)[x] for x in labels])

    ### In all cases, explanation starts here




    ## 4.