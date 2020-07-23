from scipy.cluster.hierarchy import linkage, to_tree
import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial.distance import pdist
from sklearn import linear_model
from utility import pick_cluster, path_from_root, load_data
import time

iris = load_iris()
X = iris.data
print("--- IRIS ---")

to_merge = linkage(X, method='single')
rootnode, nodelist = to_tree(to_merge, rd=True)

l = path_from_root(rootnode, nodelist[42])

counts = map(lambda n: n.count, l)
counts = list(filter(lambda c: c > 10, counts))
innertia = []

for i in range(len(counts)-1):
    innertia.append(float(counts[i])/counts[i+1])
    print(float(counts[i])/counts[i+1], end=" ")
print()

innertia = np.array(innertia)

print("counts", counts)
print("cluster", list(sorted(pick_cluster(l))))

dists = map(lambda n: n.dist, l)
dists = list(dists)

print("distances", dists)

# real data
print("--- REAL DATA ---")
MOVIE_ID = 563
U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data()

# build interp space with lasso
other_movies = list(range(all_actual_ratings.shape[1]))
other_movies.remove(MOVIE_ID)
LX = np.nan_to_num(all_actual_ratings[:, other_movies])
Ly = np.nan_to_num(all_user_predicted_ratings[:, MOVIE_ID])

# LARS learning
# complexity of the model is defined by N_FEATS: 30 features + intercept
reg = linear_model.Lars(fit_intercept=True, n_nonzero_coefs=10)
reg.fit(LX, Ly)

# select indexes of non 0 coefficients to determine the reduced space
indexes = list(np.argwhere(reg.coef_ != 0).T.flatten())
n_dim_int = np.sum(reg.coef_ != 0)

# build distance matrix for clustering
metric_indexes = indexes + [MOVIE_ID]
w = np.ones(len(metric_indexes))
w[len(indexes)] = float(2)
dist = pdist(np.nan_to_num(all_actual_ratings)[:, metric_indexes], 'cosine', w=w)

linked = linkage(dist, 'complete')
rootnode, nodelist = to_tree(linked, rd=True)

path = path_from_root(rootnode, nodelist[12])
cluster_ids = list(sorted(pick_cluster(path)))

print(len(cluster_ids))
print(cluster_ids)

dists = map(lambda n: n.dist, path)
dists = list(dists)
print(dists)