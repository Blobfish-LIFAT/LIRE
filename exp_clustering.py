import torch
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
from models import LinearRecommender, train, get_OOS_pred, get_fx
from utility import load_data, perturbations, path_from_root, pick_cluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, to_tree
import loss

# Gestion du mode pytorch CPU/GPU
from config import Config

Config.set_device_cpu()
device = Config.getInstance().device_
print("Running tensor computations on", device)

U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data()

CLS_MOVIE_WEIGHT = 2

for MOVIE_ID in [12, 954, 231, 45, 69]:
    print(" --- movie id", MOVIE_ID, "--- ")
    # build interp space with lasso
    other_movies = list(range(all_actual_ratings.shape[1]))
    other_movies.remove(MOVIE_ID)
    LX = np.nan_to_num(all_actual_ratings[:, other_movies])
    Ly = np.nan_to_num(all_user_predicted_ratings[:, MOVIE_ID])

    # LARS learning
    # complexity of the model is defined by N_FEATS: 30 features + intercept
    reg = linear_model.Lars(fit_intercept=True, n_nonzero_coefs=15)
    reg.fit(LX, Ly)

    # select indexes of non 0 coefficients to determine the reduced space
    indexes = list(np.argwhere(reg.coef_ != 0).T.flatten())
    n_dim_int = np.sum(reg.coef_ != 0)

    # build distance matrix for clustering
    # TODO -> add current film with higher wheight
    metric_indexes = indexes + [MOVIE_ID]
    w = np.ones(len(metric_indexes))
    w[len(indexes)] = float(CLS_MOVIE_WEIGHT)
    dist = pdist(np.nan_to_num(all_actual_ratings)[:, metric_indexes], 'wminkowski', p=2, w=w)


    linked = linkage(dist, 'ward')

    #plt.figure(figsize=(20, 10))
    #dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, truncate_mode='level', p=10)
    #plt.show()

    rootnode, nodelist = to_tree(linked, rd=True)

    for USER_ID in [75]:#[75,158,231,459]:

        path = path_from_root(rootnode, nodelist[USER_ID])
        cluster_ids = pick_cluster(path)
        cluster_ids.remove(USER_ID)

        print("user id", USER_ID, "cluster size", len(cluster_ids))

        n_neighbors = len(cluster_ids)

        s = torch.tensor(sigma, device=device, dtype=torch.float32)
        v = torch.tensor(Vt, device=device, dtype=torch.float32)

        # 1. Generate perturbations in interpretable space
        base_user = torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID, other_movies]), device=device, dtype=torch.float32)
        # here the interpretable space is a reduction of the initial space based on indexes
        base_user_int = base_user[reg.coef_ != 0]

        pert_int = perturbations(base_user_int, n_neighbors, std=2)
        pert_orr = torch.zeros(n_neighbors, films_nb, device=device)

        # 2. generate perturbations in original space
        i = 0
        for pu in pert_int:
            pert_orr[i] = torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID]), device=device, dtype=torch.float32)
            j = 0
            for index in indexes:
                pert_orr[i][index] = pert_int[i][j]
                j += 1
            i += 1

        y_orr = get_OOS_pred(pert_orr, s, v, films_nb)

        # add points from cluster
        base_cluster = torch.tensor(np.nan_to_num(all_actual_ratings[:, other_movies])[cluster_ids], device=device, dtype=torch.float32)
        pert_int = torch.cat((pert_int, base_cluster[:, reg.coef_ != 0]))
        pert_orr = torch.cat((pert_orr, torch.tensor(np.nan_to_num(all_actual_ratings)[cluster_ids], device=device, dtype=torch.float32)))
        y_orr = torch.cat((y_orr, torch.tensor(np.nan_to_num(all_user_predicted_ratings[cluster_ids]), device=device, dtype=torch.float32)))

        for i in range(10):
            model = LinearRecommender(n_dim_int)

            l = loss.LocalLossMAE_v3(torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID]), device=device, dtype=torch.float32), map_fn=lambda _: pert_orr, sigma=0.3)

            train(model, pert_int, y_orr[:, MOVIE_ID], l, 100, verbose=False)

            gx_ = model(base_user_int).item()
            fx = get_fx(torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID]), device=device, dtype=torch.float32), s, v, films_nb)[0, MOVIE_ID].item()
            print("mae", abs(gx_ - fx))

