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
import numpy as np
import os.path
import pickle
import umap
from scipy.sparse.linalg import svds
import hdbscan
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.spatial.distance import cosine as cosine_dist
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch
from torch import nn, optim
import datetime

from OOSPredictors import OOS_pred_slice, OOS_pred_smart
from models import LinearRecommender, train
from config import Config
from utility import read_sparse
from loss import LocalLossMAE_v3 as LocalLoss


def make_tensor(array):
    return torch.tensor(array, device=Config.device(), dtype=Config.precision())


def perturbations_gaussian(original_user, fake_users: int, std=0.47, proba=0.1):

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
    return np.clip(users, 0., 5.)


def make_black_box_slice(U, sigma, Vt, means, indexes):
    return (U[indexes] @ sigma @ Vt) + np.tile(means[indexes].reshape(len(indexes), 1), (1, Vt.shape[1]))


def explain(user_id:int, item_id:int, n_coeff:int, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size:int, pert_ratio:float=0.5, mode:str= "lime"):
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
        #Make the predictions for those
        #for k in range(pert_nb):
        #    y_train[k] = OOS_pred_smart(torch.tensor(X_train[k], device=device, dtype=torch_precision), sigma_t, Vt_t, U[user_id])[item_id].item()
        y_train[range(pert_nb)] = OOS_pred_slice(make_tensor(X_train[range(pert_nb)]), sigma_t, Vt_t).cpu().numpy()[:, item_id]

    if cluster_nb > 0:
        # generate neighbors training set part
        cluster_index = cluster_labels[user_id]
        # retrieve the cluster index of user "user_id"
        neighbors_index = np.where(cluster_labels == cluster_index)[0]
        neighbors_index = neighbors_index[neighbors_index != user_id]
        neighbors_index = np.random.choice(neighbors_index, cluster_nb)
        X_train[pert_nb:train_set_size, :] = all_user_ratings[neighbors_index, :]
        X_train[pert_nb:train_set_size, item_id] = 0

        predictor_slice = make_black_box_slice(U, sigma, Vt, user_means, neighbors_index)
        y_train[pert_nb:train_set_size] = predictor_slice[:, item_id]

    # 2. Now run a LARS linear regression model on the train set to generate the most parcimonious explanation
    if mode == "lars":
        reg = linear_model.Lars(fit_intercept=False, n_nonzero_coefs=n_coeff)
        reg.fit(X_train, y_train)
        coef = reg.coef_
        # Predict the value with the surrogate
        X_user_id = all_user_ratings[user_id]
        X_user_id[item_id] = 0
        pred = reg.predict(X_user_id.reshape(1, -1))
    # Or a classic lime style regression
    elif mode == "lime":
        model = LinearRecommender(X_train.shape[1])
        local_loss = LocalLoss(make_tensor(all_user_ratings[user_id]), sigma=5.)
        train(model, make_tensor(X_train), make_tensor(y_train), local_loss, 100, verbose=False)
        coef = model.omega.detach().cpu().numpy()
        pred = model(make_tensor(all_user_ratings[user_id])).item()
    else:
        raise NotImplementedError("No mode " + mode + " exists !")

    # Check the real prediction to get a fidelity estimation
    y_predictor_slice = make_black_box_slice(U, sigma, Vt, user_means, np.array([user_id]))
    y_predictor_slice = y_predictor_slice.transpose()[item_id]
    return coef, abs(pred - y_predictor_slice)[0]     # todo: check that in all cases reg.coef_.length is equal to # items + 1


def robustness_score(user_id, item_id, n_coeff, sigma, Vt, all_user_ratings, cluster_labels, train_set_size, pert_ratio=0.5, k_neighbors=15):

    base_exp, mae = explain(user_id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size, pert_ratio)

    # get user_id cluster neighbors
    cluster_index = cluster_labels[user_id]  # retrieve the cluster index of user "user_id"
    neighbors_index = np.where(cluster_labels == cluster_index)[0]
    neighbors_index = neighbors_index[neighbors_index != user_id]
    neighbors_index = np.random.choice(neighbors_index, k_neighbors)

    # todo: check => added a distance to all neighbors table to use in LIME-RS robustness computation
    dist_to_neighbors = np.zeros(np.shape(neighbors_index))
    # robustness
    robustness = np.zeros(15)

    cpt = 0
    for id in neighbors_index:
        # todo: here we can retrieve the absolute error on neighbors
        exp_id, _ = explain(id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size, pert_ratio)
        # todo: check => added distance into buffer
        dist_to_neighbors[cpt] = cosine_dist(all_user_ratings[user_id], all_user_ratings[id])
        robustness[cpt] = cosine_dist(exp_id, base_exp) / dist_to_neighbors[cpt]
        cpt = cpt + 1

        # todo: export

    return np.max(robustness), mae, neighbors_index, dist_to_neighbors


def robustness_score_tab(user_id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size, pert_ratio=0.5, k_neighbors=[5, 10, 15]):

    base_exp, mae = explain(user_id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size, pert_ratio)
    max_neighbors = np.max(k_neighbors)
    # get user_id cluster neighbors
    cluster_index = cluster_labels[user_id]  # retrieve the cluster index of user "user_id"
    neighbors_index = np.where(cluster_labels == cluster_index)[0]
    neighbors_index = neighbors_index[neighbors_index != user_id]
    neighbors_index = np.random.choice(neighbors_index, max_neighbors)      # look for max # of neighbors

    # objective is now to compute several robustness score for different values of k in k-NN
    dist_to_neighbors = {}      # structure to sort neighbors based on their increasing distance to user_id
    rob_to_neighbors = {}       # structure that contain the local "robustness" score of each neighbor to user_id
    for id in neighbors_index:
        # todo: here we can retrieve the absolute error on neighbors
        exp_id, _ = explain(id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size, pert_ratio)
        if (isinstance(all_user_ratings, scipy.sparse.csr._cs_matrix)):
            dist_to_neighbors[id] = cosine_dist(np.nan_to_num(all_user_ratings[user_id].toarray()), np.nan_to_num(all_user_ratings[id].toarray()))
        else:
            dist_to_neighbors[id] = cosine_dist(np.nan_to_num(all_user_ratings[user_id]), np.nan_to_num(all_user_ratings[id]))
        rob_to_neighbors[id] = cosine_dist(exp_id, base_exp) / dist_to_neighbors[id]
        if np.isnan(cosine_dist(exp_id, base_exp)):
            print('error on explanations distance')

    # sort dict values by preserving key-value relation
    sorted_dict = {k: v for k, v in sorted(dist_to_neighbors.items(), key=lambda item: item[1])} # need Python 3.6

    sorted_dist = np.zeros(max_neighbors)       # all sorted distances to user_id
    sorted_rob = np.zeros(max_neighbors)        # all robustness to user_id explanation corresponding
                                                # to sorted distance value
                                                # at index i, sorted_dist contains the i+1th distance to user_id
                                                # that corresponds to id = key
                                                # in this case, sorted_rob[i] contains robustness of id = key
    cpt = 0
    for key in sorted_dict.keys():              # checked! Keys respect the order of elements in dict
        sorted_dist[cpt] = sorted_dict[key]
        sorted_rob[cpt] = rob_to_neighbors[key]
        cpt += 1

    # finally, we compute the max(rob)@5,10,15 or any number of neighbors specified in k_neigbors
    res = np.empty(len(k_neighbors))
    cpt = 0
    for k in k_neighbors:
        res[cpt] = np.max(sorted_rob[0:k])
        cpt += 1

    # todo : check output
    return res, mae, sorted_dict.keys(), sorted_dist


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
            reducer = umap.UMAP(n_components=n_dim, n_neighbors=n_neighbor, random_state=12,
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


def experiment(U, sigma, Vt, user_means, labels, all_actual_ratings, training_set_sizes=[100], pratio=[0., 0.5, 1.0], k_neighbors=[5,10,15], n_dim_UMAP=[3, 5, 10]):
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

    USER_IDS = np.random.choice(range(U.shape[0]), 10)

    with open(OUTFILE, mode="w") as file:
        file.write('uid,iid,train_size,pert_ratio,'+",".join(["robustness_" + str(k) for k in k_neighbors])+',mae,neighbors,dist_to_neigh\n')

    for user_id in USER_IDS:
        print("[Progress] User", user_id)
        bb_slice = make_black_box_slice(U, sigma, Vt, user_means, [user_id])
        MOVIE_IDS = np.random.choice(np.argwhere(bb_slice[0] >= 0.1).flatten(), 10) # predicting 0 is too easy and the explanation can only be trivial
        for movie_id in MOVIE_IDS:
            print("[Progress] Movie", movie_id)
            out_lines = []
            for train in training_set_sizes:
                for pr in pratio:
                    # todo: check => retrieve neighbors and dist_to_neigbors and save them into output file
                    res, mae, neighbors, dist_to_neighbors = robustness_score_tab(user_id, movie_id, 10, sigma, Vt, user_means, all_actual_ratings, labels, train, pert_ratio=pr, k_neighbors=k_neighbors)
                    out = [user_id, movie_id, train, pr]
                    out.extend([r for r in res])
                    out.extend([mae, ";".join([str(n) for n in neighbors]), ";".join([str(d)for d in dist_to_neighbors])])
                    print(out) # TODO: check csv writer for file output
                    out_lines.append(",".join(map(str, out)) + "\n")
            with open(OUTFILE, mode="a+") as file:
                file.writelines(out_lines)
                out_lines.clear()


if __name__ == '__main__':
    U = None
    sigma = None
    Vt = None
    all_user_predicted_ratings = None
    OUTFILE = "res/edbt/exp_edbt_"+datetime.datetime.now().strftime("%j_%H_%M")+".csv"

    print('--- Configuring Torch')
    Config.set_device_gpu()
    print("Running tensor computations on", Config.device())

    print("--- Loading Ratings ---")
    all_actual_ratings, iid_map = read_sparse("./ml-latest-small/ratings.csv")

    # Loading data and setting all matrices
    if os.path.isfile("temp/U.gz") and os.path.isfile("temp/sigma.gz") and os.path.isfile("temp/Vt.gz") and os.path.isfile(
            "temp/labels.gz") and os.path.isfile('temp/user_means.gz'):
        print("-- LOAD MODE ---")
        U = np.loadtxt("temp/U.gz")
        sigma = np.loadtxt("temp/sigma.gz")
        Vt = np.loadtxt("temp/Vt.gz")
        labels = np.loadtxt("temp/labels.gz")
        user_means = np.loadtxt('temp/user_means.gz')
        iid_map = pickle.load(open("temp/iid_map.p", "rb"))

    # No data found computing black box and clusters results
    else:
        print('--- COMPUTE MODE ---')
        print("  De-Mean")
        user_means = all_actual_ratings.mean(axis=1)
        all_actual_ratings_demean = all_actual_ratings.todok(copy=True)
        for line, col in tqdm(all_actual_ratings_demean.keys()):
            all_actual_ratings_demean[(line, col)] -= user_means[line]

        print("  Running SVD")
        U, sigma, Vt = svds(all_actual_ratings_demean.tocsr(), k=20)
        sigma = np.diag(sigma)

        # saving matrices
        np.savetxt("temp/U.gz", U)
        np.savetxt("temp/sigma.gz", sigma)
        np.savetxt("temp/Vt.gz", Vt)
        np.savetxt('temp/user_means.gz', user_means)
        user_means = np.loadtxt('temp/user_means.gz')# Dirty fix to avoid a shape issue
        pickle.dump(iid_map, open("temp/iid_map.p", "wb"))

        print("Running UMAP")
        reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.001, metric='cosine', low_memory=True)  # metric='cosine'
        embedding = reducer.fit_transform(all_actual_ratings)
        print("Running clustering")
        clusterer = KMeans(n_clusters=5)
        clusterer.fit(embedding)
        labels = clusterer.labels_
        np.savetxt("temp/labels.gz", labels)

    if Vt.shape[1] < 10000:
        print("[WARNING] Using 100K SMALL dataset !")

    # Load sigma and Vt in memory for torch (possibly on the GPU)
    sigma_t = make_tensor(sigma)
    Vt_t = make_tensor(Vt)

    experiment(U, sigma, Vt, user_means, labels, all_actual_ratings)
"""
    import random
    from models import get_OOS_pred_single, OOS_pred_smart

    to_test = random.sample(all_actual_ratings.todok().keys(), 10)
    mae = []
    for u, i in to_test:
        mae.append(abs(OOS_pred_smart(torch.tensor(all_actual_ratings[u].toarray(), dtype=torch_precision, device=device), sigma_t, Vt_t, U[u])[i].cpu() - all_actual_ratings[u,i]))
    print("mae", np.mean(mae), np.std(mae))

    std = np.std(all_actual_ratings.toarray(), axis=0)
    print(std.shape)
    sns.distplot(std)
    plt.show()
    print(np.std(all_actual_ratings.toarray()))
"""

