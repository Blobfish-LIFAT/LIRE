import torch
import numpy as np
from sklearn import linear_model
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
from models import LinearRecommender, get_OOS_pred_inner, train, get_OOS_pred
from utility import load_data_small, perturbations, path_from_root, pick_cluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
import loss

# Gestion du mode pytorch CPU/GPU
from config import Config

Config.set_device_cpu()
device = Config.getInstance().device_
print("Running tensor computations on", device)

ratings_df = pd.read_csv("./ml-latest-small/ratings.csv")
movies_df = pd.read_csv("./ml-latest-small/movies.csv")

ratings_df = ratings_df.drop(columns=['timestamp'])
films_nb = len(set(ratings_df.movieId))

R_df = ratings_df.astype(pd.SparseDtype(np.float32, np.nan)).pivot(index='userId', columns='movieId',
                                                                   values='rating')
users_mean = R_df.mean(axis=1).values

R_demeaned = R_df.sub(R_df.mean(axis=1), axis=0)
R_demeaned = coo_matrix(R_demeaned.fillna(0).values)

U, sigma, Vt = svds(R_demeaned, k=20)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + users_mean.reshape(-1, 1)

all_actual_ratings = R_df.values
cond = np.invert(np.isnan(all_actual_ratings))
np.copyto(all_user_predicted_ratings, all_actual_ratings, where=cond)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



movie_id = 350

# build interp space with lasso
other_movies = list(range(all_actual_ratings.shape[1]))
other_movies.remove(movie_id)
LX = np.nan_to_num(all_actual_ratings[:, other_movies])
Ly = np.nan_to_num(all_user_predicted_ratings[:, movie_id])

# LARS learning
# complexity of the model is defined by N_FEATS: 30 features + intercept

reg = linear_model.Lars(fit_intercept=True, n_nonzero_coefs=15)
reg.fit(LX, Ly)

# select indexes of non 0 coefficients to determine the reduced space
indexes = np.argwhere(reg.coef_ != 0).T.flatten()
n_dim_int = np.sum(reg.coef_ != 0)

# build distance matrix for clustering
# TODO -> add current film with higher wheight
dist = pdist(np.nan_to_num(all_actual_ratings)[:, list(indexes)])


linked = linkage(dist, 'ward')

#plt.figure(figsize=(20, 10))
#dendrogram(linked, orientation='top', distance_sort='descending',
#            show_leaf_counts=True, truncate_mode='level', p=10)
#plt.show()

rootnode, nodelist = to_tree(linked, rd=True)

user_id = 42

path = path_from_root(rootnode, nodelist[user_id])
cluster_ids = pick_cluster(path)
cluster_ids.remove(user_id)
print(cluster_ids)

n_neighbors = len(cluster_ids)

s = torch.tensor(sigma, device=device, dtype=torch.float32)
v = torch.tensor(Vt, device=device, dtype=torch.float32)

# 1. Generate perturbations in interpretable space
base_user = torch.tensor(np.nan_to_num(all_actual_ratings[user_id, other_movies]), device=device, dtype=torch.float32)
# here the interpretable space is a reduction of the initial space based on indexes
base_user_int = base_user[reg.coef_ != 0]

pert_int = perturbations(base_user_int, n_neighbors, std=2)
pert_orr = torch.zeros(n_neighbors, films_nb - 1, device=device)

# 2. generate perturbations in original space
i = 0
for pu in pert_int:
    pert_orr[i] = base_user.detach().clone()
    j = 0
    for index in indexes:
        pert_orr[i][index] = pert_int[i][j]
        j += 1
    i += 1

print(pert_orr.size(), s.size(), v.size(), films_nb)
y_orr = get_OOS_pred(pert_orr, s, v, films_nb)

# add points from cluster
base_cluster = torch.tensor(np.nan_to_num(all_actual_ratings[:, other_movies])[cluster_ids], device=device, dtype=torch.float32)
pert_int = torch.cat((pert_int, base_cluster[:, reg.coef_ != 0]))
pert_orr = torch.cat((pert_orr, base_cluster))

print(y_orr.size())

model = LinearRecommender(n_dim_int)

l = loss.LocalLossMAE_v3(base_user, map_fn=lambda _: pert_orr, sigma=0.3)

train(model, pert_int, y_orr[:, movie_id], l, 100, verbose=False)

gx_ = model(base_user_int).item()
fx = y_orr[:, movie_id].mean().item()
print(gx_, fx, all_user_predicted_ratings[user_id, movie_id])


# -> lasso pour le film
# -> id cluster de l'user
# -> trouver la coupure, regarder liste d'agregation de l'user et taille des clusters specific faire le cut d'inertie max
# -> do smart perturbations from user, from cluster points, etc