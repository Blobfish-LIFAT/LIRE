import torch
import numpy as np
from sklearn import linear_model
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
from models import LinearRecommender, get_OOS_pred_inner, train
from utility import load_data_small, perturbations
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

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

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

# build distance matrix for clustering
dist = pdist(np.nan_to_num(all_actual_ratings)[:, list(indexes)])
print(dist)

#c_dist = pdist(np.nan_to_num(all_actual_ratings), metric='cosine')
#dist = pdist(np.nan_to_num(all_actual_ratings)[:, [movie_id]])

#dist += (c_dist/2.5)
#del c_dist

linked = linkage(dist, 'ward')

plt.figure(figsize=(20, 10))
dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True,
           truncate_mode='level',
           p=10)
plt.show()

#clustering = AgglomerativeClustering(affinity='precomputed', linkage='average', distance_threshold=0, n_clusters=None)
#clustering.fit(squareform(dist))

#print(squareform(dist))

#plot_dendrogram(clustering, truncate_mode='level', p=3)
#plt.show()


# -> lasso pour le film
# -> id cluster de l'user
#