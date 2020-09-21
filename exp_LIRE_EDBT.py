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
from utility import load_data as load_data
from scipy.optimize import least_squares
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

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

def perturbations_gaussian(original_user, fake_users: int, std=2, proba=0.1):
    ux = original_user.toarray()# Comes from a scipy sparse matrix
    nb_dim = ux.shape[1]
    users = np.tile(ux, (fake_users, 1))

    noise = np.random.normal(0, std, (fake_users, nb_dim))
    rd_mask = np.random.binomial(1, proba, (fake_users, nb_dim))
    noise = noise * rd_mask * (users != 0.)
    users = users + noise
    # users[users > 5.] = 5. #seems to introduce detrimental bias in the training set
    return np.clip(users, 0., None)


def explain(user_id, item_id, n_coeff, sigma, Vt, all_user_ratings, cluster_labels, train_set_size, pert_ratio=0.5):
    """

    :param user_id: user for which an explanation is expected
    :param item_id: item for which an explanation is expected
    :param n_coeff: number of coefficients for the explanation
    :param sigma: intensity of latent factors
    :param Vt: item latent space
    :param all_user_ratings: matrix containing all predicted user ratings
    :param cluster_labels: vector indicating for each user its cluster label
    :param train_set_size: size of the train set (perturbation + neighbors from clusters) to train local surrogate
    :param pert_ratio: perturbation ratio
    :return: a vector representing
    """

    # 1. Generate a train set for local surrogate model
    X_train = np.empty(0)
    y_train = np.empty(0)

    # todo: remove item_id from the user vector !!!

    pert_nb = int(train_set_size * pert_ratio)      # nb of perturbed entries
    cluster_nb = train_set_size - pert_nb           # nb of real neighbors

    if pert_nb > 0:                                 # generate perturbed training set part
        # generate perturbed users
        X_train = perturbations_gaussian(all_user_ratings[user_id],pert_nb)
        ## do the oos prediction
        for p in X_train:
            np.append(y_train, oos_predictor(p, sigma, Vt))

    if cluster_nb > 0:                              # generate neighbors training set part
        cluster_index = cluster_labels[user_id]             # retrieve the cluster index of user "user_id"
        neighbors_index = np.where(cluster_labels == cluster_index)
        neighbors_index = np.random.choice(neighbors_index, cluster_nb)
        np.append(X_train, all_user_ratings[neighbors_index])   # todo: remove item_id from the user vector !!!
        np.append(y_train, all_user_ratings[neighbors_index,item_id])

    # 2. Now run a LARS linear regression model on the train set to generate the most parcimonious explanation
    reg = linear_model.Lars(n_nonzero_coefs= n_coeff)
    reg.fit(X_train, y_train)               # todo add here the training set, check the format
    return reg.coef_        # todo: check that the intercept is not in the set of coeff
                            # as it is not related to an item feature


## code
if __name__ == '__main__':
    U = None
    sigma = None
    Vt = None
    all_user_predicted_ratings = None

    # 1. Loading data and setting all matrices
    if os.path.isfile("U.gz") and os.path.isfile("sigma.gz") and os.path.isfile("Vt.gz") \
            and os.path.isfile("all_ratings.gz") and os.path.isfile("labels.gz"):
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

    print("Running UMAP")
    ## 2. Determining neighborhood as a clustering problem
    from utility import read_sparse
    if not ("all_actual_ratings" in locals() or "all_actual_ratings" in globals()):
        all_actual_ratings = read_sparse("./ml-latest-small/ratings.csv")
    reducer = umap.UMAP(n_components=3, n_neighbors=20, random_state=12)  # metric='cosine'
    embedding = reducer.fit_transform(np.nan_to_num(all_actual_ratings))

    print("Running clustering")
    ## 3. Clustering
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(embedding)
    labels = clusterer.labels_
    np.savetxt("labels.gz", labels)

    # visualization of neighborhood
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
               c=[sns.color_palette("husl", max(labels) + 2)[x] for x in labels])
    plt.show()


    ### In all cases, explanation starts here
    ## 4.
    uid = 42
    iid = 69
    explain(uid, iid, 10, sigma, Vt, all_actual_ratings, labels, 50, 0.05)