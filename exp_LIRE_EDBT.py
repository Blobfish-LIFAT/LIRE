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
from utility import load_data_small as load_data_small
from scipy.optimize import least_squares
import hdbscan
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from scipy.spatial.distance import cosine as cosine_dist

## constants
VERBOSE = False
F_LATENT_FACTORS = 10
DIM_UMAP = 3            # number of dimensions after UMAP reduction
N_TRAINING_POINTS = 50
PERT_STD = 1.04 # empiricaly determined from dataset
# MOVIE_IDS = random.sample(range(2000), 10)
# USER_IDS = random.sample(range(610), 10)

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

    if(isinstance(original_user,scipy.sparse.csr.csr_matrix)):
        original_user = original_user.toarray()
    else:
        original_user = original_user.reshape(1,len(original_user))
    # Comes from a scipy sparse matrix
    nb_dim = original_user.shape[1]
    users = np.tile(original_user, (fake_users, 1))

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
    #print(type(all_user_ratings))
    #t = isinstance(all_user_ratings,scipy.sparse.csr.csr_matrix)
    if(isinstance(all_user_ratings,scipy.sparse.csr.csr_matrix)):
        all_user_ratings = all_user_ratings.toarray()
    else:
        all_user_ratings = np.nan_to_num(all_user_ratings)

    # 1. Generate a train set for local surrogate model
    X_train = np.zeros((train_set_size, Vt.shape[1]))   # 2D array
    y_train = np.zeros(train_set_size)                  # 1D array

    pert_nb = int(train_set_size * pert_ratio)      # nb of perturbed entries
    cluster_nb = train_set_size - pert_nb           # nb of real neighbors
    if pert_nb > 0:                                 # generate perturbed training set part
        # generate perturbed users
        X_train[0:pert_nb, :] = perturbations_gaussian(all_user_ratings[user_id], pert_nb)
        X_train[0:pert_nb, item_id] = 0
        # todo: check if correct CHECKED
        ## do the oos prediction
        for i in range(pert_nb):
            y_train[i] = oos_predictor(X_train[i], sigma, Vt)[item_id]

    if cluster_nb > 0:
        # generate neighbors training set part
        cluster_index = cluster_labels[user_id]
        # retrieve the cluster index of user "user_id"
        neighbors_index = np.where(cluster_labels == cluster_index)[0]
        # todo: check [0] CHECKED
        neighbors_index = np.random.choice(neighbors_index, cluster_nb)
        t = all_user_ratings[neighbors_index, :]
        X_train[pert_nb:train_set_size, :] = all_user_ratings[neighbors_index,:]
        X_train[pert_nb:train_set_size, item_id] = 0
        # todo: check if correct => CHECKED

        y_train[pert_nb:train_set_size] = all_user_ratings[neighbors_index, item_id]

    # 2. Now run a LARS linear regression model on the train set to generate the most parcimonious explanation
    reg = linear_model.Lars(fit_intercept=True, n_nonzero_coefs= n_coeff)
    reg.fit(X_train, y_train)
    # todo: check item_id to explain is 0 CHECKED
    return reg.coef_       # todo: check that in all cases reg.coef_.length is equal to # items + 1


def robustness_score(user_id, item_id, n_coeff, sigma, Vt, all_user_ratings, cluster_labels, train_set_size, pert_ratio=0.5, k_neighbors=15):

    base_exp = explain(user_id, item_id, n_coeff, sigma, Vt, all_user_ratings, cluster_labels, train_set_size, pert_ratio)

    # get user_id cluster neighbors
    cluster_index = cluster_labels[user_id]  # retrieve the cluster index of user "user_id"
    neighbors_index = np.where(cluster_labels == cluster_index)[0]
    neighbors_index = np.random.choice(neighbors_index, k_neighbors)

    # robustness
    robustness = np.zeros(15)

    cpt = 0
    for id in neighbors_index:
        exp_id = explain(id, item_id, n_coeff, sigma, Vt, all_user_ratings, cluster_labels, train_set_size, pert_ratio)
        robustness[cpt] = cosine_dist(exp_id, base_exp) / cosine_dist(all_user_ratings[user_id], all_user_ratings[id])
        cpt = cpt + 1

    return np.max(robustness)

def robustness_score_tab(user_id, item_id, n_coeff, sigma, Vt,
                         all_user_ratings, cluster_labels, train_set_size,
                         pert_ratio=0.5, k_neighbors=[5, 10, 15]):

    base_exp = explain(user_id, item_id, n_coeff, sigma, Vt, all_user_ratings, cluster_labels, train_set_size, pert_ratio)
    max_neighbors = np.max(k_neighbors)
    # get user_id cluster neighbors
    cluster_index = cluster_labels[user_id]  # retrieve the cluster index of user "user_id"
    neighbors_index = np.where(cluster_labels == cluster_index)[0]
    neighbors_index = np.random.choice(neighbors_index, max_neighbors)      # look for max # of neighbors

    # objective is now to compute several robustness score for different values of k in k-NN
    dist_to_neighbors = {}      # structure to sort neighbors based on their increasing distance to user_id
    rob_to_neighbors = {}       # structure that contain the local "robustness" score of each neighbor to user_id
    for id in neighbors_index:
        exp_id = explain(id, item_id, n_coeff, sigma, Vt, all_user_ratings, cluster_labels, train_set_size, pert_ratio)
        dist_to_neighbors[id] = cosine_dist(all_user_ratings[user_id], all_user_ratings[id])
        rob_to_neighbors[id] = cosine_dist(exp_id, base_exp) / dist_to_neighbors[id]

    # sort dict values by preserving key-value relation
    sorted_dict = {k: v for k, v in sorted(dist_to_neighbors.items(), key=lambda item: item[1])} # need Python 3.6

    sorted_dist = np.zeros(max_neighbors)       # all sorted distances to user_id
    sorted_rob = np.zeros(max_neighbors)        # all robustness to user_id explanation corresponding
                                                # to sorted distance value
                                                # at index i, sorted_dist contains the i+1th distance to user_id
                                                # that corresponds to id = key
                                                # in this case, sorted_rob[i] contains robustness of id = key

    cpt = 0
    for key in sorted_dict.keys():              # todo: check that keys respect the order of elements in dict
        sorted_dist[cpt] = sorted_dict[key]
        sorted_rob[cpt] = rob_to_neighbors[key]
        cpt += 1

    # finally, we compute the max(rob)@5,10,15 or any number of neighbors specified in k_neigbors
    res = np.empty(len(k_neighbors))
    cpt = 0
    for k in k_neighbors:
        res[cpt] = np.max(sorted_rob[0:k])
        cpt += 1

    return res

def exp_check_UMAP(users, items, n_coeff, sigma, Vt, all_user_ratings, cluster_labels, train_set_size=50,
                   n_dim_UMAP=[3, 5, 10], n_neighbors_UMAP=[10, 30, 50], pert_ratio=0.5, k_neighbors=10):
    """
    Run test to evaluate the sensitivity of our method to UMAP dimensionality reduction
    :param users: numpy array of user ids
    :param items: numpy array of item ids
    :param n_coeff: number of coefficients of interpretable features for the explanation
    :param sigma: influence of each latent dimension
    :param Vt: latent item space
    :param all_user_ratings: initial rating matrix
    :param cluster_labels: user clustering, defines neighborhood
    :param train_set_size: number of training instance to learn surrogate model
    :param n_dim_UMAP: numpy array of reducted number of dimensions
    :param n_neighbors_UMAP: numpy array of number of neighbors to preserve local topology in UMAP
    :param training_set_size:
    :param pert_ratio:
    :param k_neighbors:
    :return:
    """
    for n_dim in n_dim_UMAP:
        for n_neighbor in n_neighbors_UMAP:
            # UMAP
            reducer = reducer = umap.UMAP(n_components=n_dim, n_neighbors=n_neighbor, random_state=12,
                                          min_dist=0.0001)  # metric='cosine'
            embedding = reducer.fit_transform(np.nan_to_num(all_actual_ratings))

            # clustering
            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(embedding)
            labels = clusterer.labels_
            np.savetxt("labels_" + str(n_dim) + "_" + str(n_neighbor) +".gz", labels)   # personalize output filename

            # robustness measure
            for user_id in users:
                for item_id in items:
                    # todo : save in a file somewhere, which format?
                    robustness_score(user_id, item_id, n_coeff, sigma, Vt, all_user_ratings,
                                     cluster_labels, train_set_size, pert_ratio, k_neighbors)





def experiment(training_set_sizes=[50, 100, 150, 200], pratio=[0., 0.5, 1.0], k_neighbors=[5,10,15], n_dim_UMAP=[3, 5, 10]):
    """
    Run the first experiment that consists in choosing randomly users and items and each time providing the robustness
    of explanation and its complexity in terms of number of non-zero features
    Parameters of the experiment are:
    :param training_set_sizes: different training set sizes to learn surrogate models
    :param pratio: different perturbation ratio in training set, modulate # of perturbed points vs # of cluster points
    :param k_neighbors: different number of neighbors to compute robustness
    :param n_dim_UMAP: size of projected space for UMAP
    :return:
    """

    for train in training_set_sizes:
        for pr in pratio:


    pass

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
    reducer = reducer = umap.UMAP(n_components=3, n_neighbors=30, random_state=12, min_dist=0.0001)    # metric='cosine'
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
    coefs = explain(uid, iid, 10, sigma, Vt, all_actual_ratings, labels, 50, 0.05)
    print(np.where(coefs >0))
