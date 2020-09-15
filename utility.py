import torch

import pandas as pd
import numpy as np

from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
from scipy.cluster.hierarchy import dendrogram

from math import sqrt, pow
from numpy.linalg import norm
from numba import jit
from perturbations import perturbations_gaussian
import time

def load_data_small():
    """
    Loads a very small subset of the data for debug purposes only
    :return: U, Sigma, Vt resutling of the SVD, along the raw user ratings and predicted user ratings and the pandas DFs
    """
    ratings_df = pd.read_csv("./ml-latest-small/ratings.csv")
    movies_df = pd.read_csv("./ml-latest-small/movies.csv")

    # Make it smaller
    #ratings_df = ratings_df[ratings_df["userId"] < 51].drop(columns=['timestamp'])
    #ratings_df = ratings_df.query("movieId < 400")
    films_nb = len(set(ratings_df.movieId))

    R_df = ratings_df.astype(pd.SparseDtype(np.float32, np.nan)).pivot(index='userId', columns='movieId',
                                                                       values='rating')
    users_mean = R_df.mean(axis=1).values

    R_demeaned = R_df.sub(R_df.mean(axis=1), axis=0)
    R_demeaned = coo_matrix(R_demeaned.fillna(0).values)

    U, sigma, Vt = svds(R_demeaned, k=10)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + users_mean.reshape(-1, 1)

    all_actual_ratings = R_df.values
    cond = np.invert(np.isnan(all_actual_ratings))
    np.copyto(all_user_predicted_ratings, all_actual_ratings, where=cond)

    return U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb


def load_data():
    start = time.time()

    ratings_df = pd.read_csv("./ml-20m/ratings.csv")
    movies_df = pd.read_csv("./ml-20m/movies.csv")
    done = time.time()
    '''rated_movies = list(set(ratings_df['movieId'].values.tolist()))
    all_movies = movies_df['movieId'].values.tolist()
    size1 = len(rated_movies)
    size2 = len(all_movies)'''

    ratings_df = ratings_df.drop(columns=['timestamp'])
    films_nb = len(set(ratings_df.movieId))
    print("Chargement des données en mémoire réalisé en :", done - start, ' secondes')
    start = time.time()
    R_df = ratings_df.astype(pd.SparseDtype(np.float32, np.nan)).pivot(index='userId', columns='movieId',
                                                                       values='rating')
    done = time.time()
    print("Pivot des données réalisé en :", done - start, ' secondes')

    raw_mean = R_df.mean(axis=1)
    users_mean = raw_mean.values

    start = time.time()
    R_demeaned = R_df.sub(raw_mean, axis=0)
    R_demeaned = coo_matrix(R_demeaned.fillna(0).values)

    U, sigma, Vt = svds(R_demeaned, k=20)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + users_mean.reshape(-1, 1)

    done = time.time()
    print("Entrainement de la black-box et réalisation des prédictions par la black box:", done - start, ' secondes')
    all_actual_ratings = R_df.values
    cond = np.invert(np.isnan(all_actual_ratings))
    #np.copyto(all_user_predicted_ratings, all_actual_ratings, where=cond)

    return U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb


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


def path_from_root(r, node):
    def hasPath_int(root, arr):
        if root is None:
            return False

        arr.append(root)

        if root is node:
            return True
        elif hasPath_int(root.left, arr) or hasPath_int(root.right, arr):
            return True
        else:
            arr.pop(-1)
            return False

    acc = []
    if hasPath_int(r, acc):
        return acc
    else:
        return None


def pick_cluster(l):
    counts = map(lambda n: n.count, l)
    counts = list(filter(lambda c: c > 10, counts))
    inertia = []

    for i in range(len(counts) - 1):
        inertia.append(float(counts[i]) / counts[i + 1])

    inertia = np.array(inertia)
    index = np.argmax(inertia) + 1

    nodes = l[index].pre_order(lambda x: x)
    nodes = filter(lambda n: n.is_leaf(), nodes)

    return list(map(lambda n: n.get_id(), nodes))


def get_user_cluster(USER_ID, root_node, user_node):
    path = path_from_root(root_node, user_node)
    try:
        cluster_ids = pick_cluster(path)
    except ValueError:
        print("[ERROR] Error on clustering")
        return None

    if USER_ID in cluster_ids:
        cluster_ids.remove(USER_ID)

    return cluster_ids


def epsilon_neighborhood_fast(R, uindx, epsilon):
    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(R, R.T)
    # inverse squared magnitude
    inv_square_mag = 1 / np.diag(similarity)
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = 1 - (cosine.T * inv_mag)
    udist = cosine[uindx]
    nmask = udist < epsilon
    nmask[uindx] = False
    #print(sum(nmask), " neighbors found (epsilon=", epsilon, ")")
    return np.argwhere(nmask)


@jit
def k_neighborhood(R, uindx, k):
    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(R, R.T)
    # inverse squared magnitude
    inv_square_mag = 1 / np.diag(similarity)
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = 1 - (cosine.T * inv_mag)
    udist = cosine[uindx]

    res = list(np.argsort(udist)[:k+1])
    res.remove(uindx)
    return res


from scipy.spatial.distance import cosine as cosine_dist

def robustness(target, target_expl, neighborhood, neighborhood_expl):
    ratios = []
    for i, neighbor in enumerate(neighborhood):
        ratio = cosine_dist(target_expl, neighborhood_expl[i]) / cosine_dist(target, neighbor)
        ratios.append(ratio)
    return max(ratios)

@jit(nopython=True, parallel=True)
def fast_similarity(R):

    similarity = np.dot(R, R.T)
    # inverse squared magnitude
    inv_square_mag = 1 / np.diag(similarity)
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = 1 - (cosine.T * inv_mag)
    return cosine


if __name__ == '__main__':

    U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data()

    R = np.nan_to_num(all_actual_ratings)

    similarity = np.dot(R, R.T)
    # inverse squared magnitude
    inv_square_mag = 1 / np.diag(similarity)
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = 1 - (cosine.T * inv_mag)

    with open("res/exp_willeme.csv", mode="w") as fout:
        fout.write("userId;voisins;distances\n")
        for user in [12, 601, 89, 154, 548, 325, 245, 45, 489, 42, 29, 215]:

            udist = cosine[user]
            res = list(np.argsort(udist)[:15 + 1])
            res.remove(user)
            print("UID", user + 1, sum(R[user]))
            print(list(map(lambda x : x + 1, res)))
            print(cosine[user, res])

            fout.write(str(user +1 ) + ";" + str(list(map(lambda x : x + 1, res))) + ";" + str(list(cosine[user, res])) + "\n")
