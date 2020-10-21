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
import pandas as pd
import os.path
import pickle
import umap
from scipy.sparse.linalg import svds
import hdbscan
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_dist
from tqdm import tqdm
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import torch
import datetime

from OOSPredictors import OOS_pred_slice
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
        original_user = original_user.reshape(1, len(original_user))
    # Comes from a scipy sparse matrix
    nb_dim = original_user.shape[1]
    users = np.tile(original_user, (fake_users, 1))

    noise = np.random.normal(np.zeros(nb_dim), global_variance/2, (fake_users, nb_dim))
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

    X_user_id = all_user_ratings[user_id].copy()
    X_user_id[item_id] = 0

    # Check the real prediction
    y_predictor_slice = make_black_box_slice(U, sigma, Vt, user_means, np.array([user_id]))
    y_predictor_slice = y_predictor_slice.transpose()[item_id]

    # 2. Now run a LARS linear regression model on the train set to generate the most parcimonious explanation
    if mode == "lars":
        reg = linear_model.Lars(fit_intercept=False, n_nonzero_coefs=n_coeff)
        reg.fit(X_train, y_train)
        coef = reg.coef_
        # Predict the value with the surrogate
        pred = reg.predict(X_user_id.reshape(1, -1))
    # Or a classic lime style regression
    elif mode == "lime":
        models_ = []
        errors_ = []
        # A few runs to avoid bad starts
        for _ in range(5):
            model = LinearRecommender(X_train.shape[1])
            local_loss = LocalLoss(make_tensor(all_user_ratings[user_id]), sigma=5., alpha=0.001)
            train(model, make_tensor(X_train), make_tensor(y_train), local_loss, 100, verbose=False)
            pred = model(make_tensor(X_user_id)).item()
            models_.append(model)
            errors_.append(abs(pred - y_predictor_slice)[0])
            models_.append(model)
            if abs(pred - y_predictor_slice)[0] < 0.1:  # Good enough
                break
        best = models_[np.argmin(errors_)]
        coef = best.omega.detach().cpu().numpy()
        pred = best(make_tensor(X_user_id)).item()
    else:
        raise NotImplementedError("No mode " + mode + " exists !")

    print("Local prediction : ",pred)
    print("Black-box prediction :", y_predictor_slice[0])
    return coef, abs(pred - y_predictor_slice)[0]


def robustness_score_tab(user_id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size:int, pert_ratio:float=0.5, k_neighbors=[5, 10, 15]):

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


def exp_check_UMAP(n_coeff, sigma, Vt, all_user_ratings, cluster_labels, train_set_size=50,n_dim_UMAP = [3,10,15],
                   min_dist_UMAP=[0.1,0.01,0.001], n_neighbors_UMAP=[10, 30, 50], pert_ratio:float=0, k_neighbors=[5, 10, 15]):
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
    items = np.random.choice(range(Vt.shape[1]), 5)
    users = np.random.choice(range(U.shape[0]), 10)
    columns = ['clustering_algorithm','n_cluster','robustness@5','robustness@10','robustness@15','mae','user_id','item_id','silhoutte_score_by_cluster','silhouette_score_all']
    result = []
    for n_dim in n_dim_UMAP:
        for min_dist in min_dist_UMAP:
            for n_neighbor in n_neighbors_UMAP:
                reducer = umap.UMAP(n_components=n_dim, n_neighbors=n_neighbor, random_state=12,
                                    min_dist=min_dist)  # metric='cosine'
                embedding = reducer.fit_transform(all_actual_ratings)
                for clustering_algorithm in ['kmeans','hdbscan']:

                    if (clustering_algorithm == 'kmeans'):

                        hyperparameters_clustering = [5,10,15]
                        for hyperparameter in hyperparameters_clustering:
                            hyperparameter_clustering = hyperparameter
                            clusterer = KMeans(n_clusters = hyperparameter)
                            clusterer.fit(embedding)
                            labels = clusterer.labels_

                            #Calcul of silhouette scores by sample, cluster and all
                            X = all_user_ratings.toarray()
                            silh_samp = silhouette_samples(X=X, labels=labels, metric='cosine')
                            silh_score = silhouette_score(X=X, labels=labels, metric='cosine')
                            df_temp = pd.DataFrame(silh_samp,columns=['silh_samp'])
                            df_temp['labels'] = labels
                            silh_clust = [df_temp.loc[df_temp['labels']==label]['silh_samp'].mean() for label in set(labels)]

                            np.savetxt("labels_" + str(n_dim) + "_" + str(n_neighbor) + ".gz",
                                       labels)  # personalize output filename

                            # robustness measure
                            for user_id in users:
                                for item_id in items:
                                    # todo : save in a file somewhere, which format?

                                    res, mae, keys, distances = robustness_score_tab(user_id, item_id, n_coeff, sigma,
                                                                                     Vt, user_means, all_user_ratings,
                                                                                     labels, train_set_size, pert_ratio,
                                                                                     k_neighbors)
                                    l = [clustering_algorithm, hyperparameter_clustering, res[0], res[1], res[2], mae,
                                         user_id, item_id, silh_score, silh_clust]
                                    print("user :", user_id, " item_id", item_id, " line added :", l)
                                    result.append(l)
                    else:
                        hyperparameter_clustering = None
                        clusterer = hdbscan.HDBSCAN()

                        clusterer.fit(embedding)

                        labels = clusterer.labels_
                        # Calcul of silhouette scores by sample, cluster and all
                        X = all_user_ratings.toarray()
                        silh_samp = silhouette_samples(X=X, labels=labels, metric='cosine')
                        silh_score = silhouette_score(X=X, labels=labels, metric='cosine')
                        df_temp = pd.DataFrame(silh_samp, columns=['silh_samp'])
                        df_temp['labels'] = labels
                        silh_clust = [df_temp.loc[df_temp['labels'] == label]['silh_samp'].mean() for label in set(labels)]

                        np.savetxt("labels_" + str(n_dim) + "_" + str(n_neighbor) +".gz", labels)   # personalize output filename

                        # robustness measure
                        for user_id in users:
                            for item_id in items:
                                # todo : save in a file somewhere, which format?
                                res,mae, keys, distances = robustness_score_tab(user_id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings,
                                                 labels, train_set_size, pert_ratio, k_neighbors)
                                l = [clustering_algorithm,hyperparameter_clustering,res[0],res[1],res[2],mae,user_id,item_id,silh_score,silh_clust]
                                print("user :",user_id," item_id",item_id, " line added :",l)
                                result.append(l)

    df = pd.DataFrame(result, columns=columns)
    df.to_csv('res/clustering_result_parameters_search_v2.csv')


def experiment_test_top_recommendation(U, sigma, Vt, user_means, labels, all_actual_ratings, training_set_sizes=[100], pratio=[0.1, 0.9],
               k_neighbors=[5, 10, 15]):
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

    n_coeff = 10
    train_set_size = 1000
    pert_ratio = 1

    toBeExplained = [(int(x[0]), int(x[1]),x[2]) for x in
                     pd.read_csv('ml-latest-small/ratings.csv', sep=",").values.tolist()[:1000]]
    for couple_uid_iid in toBeExplained:
        base_exp, mae = explain(couple_uid_iid[0], couple_uid_iid[1], n_coeff, sigma, Vt, user_means, all_actual_ratings, labels, train_set_size, pert_ratio)
        print("Mae = ", mae)
        print("True rating = ", couple_uid_iid[2])
        print("--------------------------")



def experiment(U, sigma, Vt, user_means, labels, all_actual_ratings, training_set_sizes=[100], pratio=[0.1, 0.9], k_neighbors=[5,10,15]):
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


def vars(a, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))


def stds(a, axis=None):
    """ Standard deviation of sparse matrix a
    std = sqrt(var(a))
    """
    return np.sqrt(vars(a, axis))


if __name__ == '__main__':
    U = None
    sigma = None
    Vt = None
    all_user_predicted_ratings = None
    OUTFILE = "res/edbt/exp_edbt_"+datetime.datetime.now().strftime("%j_%H_%M")+".csv"
    TEMP = "temp/"# Fodler for precomputed black box data
    SIZE = "small"#Size of dataset small/big 100k or 20M

    print('--- Configuring Torch')
    Config.set_device_gpu()
    print("Running tensor computations on", Config.device())

    print("--- Loading Ratings ---")
    if SIZE == "small":
        all_actual_ratings, iid_map = read_sparse("./ml-latest-small/ratings.csv")
        TAG = "_small_"
        print("[WARNING] Using 100K SMALL dataset !")
    else:
        all_actual_ratings, iid_map = read_sparse("./ml-20m/ratings.csv")
        TAG = ""

    # Loading data and setting all matrices
    if os.path.isfile(TEMP + "flag" + TAG + ".flag"):
        print("-- LOAD MODE ---")
        U = np.loadtxt(TEMP + "U" + TAG + ".gz")
        sigma = np.loadtxt(TEMP + "sigma" + TAG + ".gz")
        Vt = np.loadtxt(TEMP + "Vt" + TAG + ".gz")
        labels = np.loadtxt(TEMP + "labels" + TAG + ".gz")
        user_means = np.loadtxt(TEMP + "user_means" + TAG + ".gz")
        iid_map = pickle.load(open(TEMP + "iid_map" + TAG + ".p", "rb"))

    # No data found computing black box and clusters results
    else:
        print('--- COMPUTE MODE ---')
        print("  De-Mean")
        user_means = [None] * all_actual_ratings.shape[0]
        all_actual_ratings_demean = scipy.sparse.dok_matrix(all_actual_ratings.shape)
        for line, col in tqdm(all_actual_ratings.todok().keys()):
            if user_means[line] is None:
                user = all_actual_ratings[line].toarray()
                user[user == 0.] = np.nan
                user_means[line] = np.nanmean(user)
            all_actual_ratings_demean[line, col] = all_actual_ratings[line, col] - user_means[line]
        user_means = np.array(user_means)
        user_means = user_means.reshape(all_actual_ratings.shape[0],1)

        print("  Running SVD")
        U, sigma, Vt = svds(all_actual_ratings_demean.tocsr(), k=50,solver='lobpcg',which='LM', maxiter=1000)
        sigma = np.diag(sigma)

        # saving matrices
        np.savetxt(TEMP + "U" + TAG + ".gz", U)
        np.savetxt(TEMP + "sigma" + TAG + ".gz", sigma)
        np.savetxt(TEMP + "Vt" + TAG + ".gz", Vt)
        np.savetxt(TEMP + "user_means" + TAG + ".gz", user_means)
        user_means = np.loadtxt(TEMP + "user_means" + TAG + ".gz")# Dirty fix to avoid a shape issue
        pickle.dump(iid_map, open(TEMP + "iid_map" + TAG + ".p", "wb"))

        print("Running UMAP")
        reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.01, low_memory=True)  # metric='cosine'
        embedding = reducer.fit_transform(all_actual_ratings)
        print("Running clustering")
        clusterer = KMeans(n_clusters=75)
        clusterer.fit(embedding)
        labels = clusterer.labels_
        np.savetxt(TEMP + "labels" + TAG + ".gz", labels)
        with open(TEMP + "flag" + TAG + ".flag", mode="w") as f:
            f.write("1")


    # Load sigma and Vt in memory for torch (possibly on the GPU)
    sigma_t = make_tensor(sigma)
    Vt_t = make_tensor(Vt)

    global_variance = stds(all_actual_ratings, axis=0)

    experiment_test_top_recommendation(U, sigma, Vt, user_means, labels, all_actual_ratings)
    #exp_check_UMAP(10, sigma, Vt, all_actual_ratings, None, train_set_size=50, n_dim_UMAP=[3],
    #                   min_dist_UMAP=[0.01], n_neighbors_UMAP=[30], pert_ratio=0., k_neighbors=[5, 10, 15])


