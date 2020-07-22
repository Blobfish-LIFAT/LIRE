import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
from scipy.cluster.hierarchy import dendrogram
from random import sample

from config import Config


# Make Perturbations
def perturbations(ux, fake_users, std=2, proba=0.1):
    nb_dim = ux.size()[0]
    users = ux.expand(fake_users, nb_dim)

    perturbation = nn.init.normal_(torch.zeros(fake_users, nb_dim, device=Config.device()), 0, std)
    rd_mask = torch.zeros(fake_users, nb_dim, device=Config.device()).uniform_() > (1. - proba)
    perturbation = perturbation * rd_mask * (users != 0.)
    users = users + perturbation
    # users[users > 5.] = 5. #seems to introduce detrimental bias in the training set
    return torch.abs(users)


# Make Perturbations
def perturbations_uniform(ux, fake_users, proba=0.25):
    """
    :param ux: initial user
    :param fake_users: number of perturbed copies from the initial user
    :param proba: probability of changing a value
    :return: a matrix of perturbed fake users
    """
    nb_dim = ux.size()[0]
    users = ux.expand(fake_users, nb_dim)
    rd_mask = torch.zeros(fake_users, nb_dim, device=Config.device()).uniform_() > (proba)

    #print(rd_mask * users)

    return rd_mask * users

# Make Perturbations
def perturbations_3(ux):
    """
        Generate all possible perturbed users from ux by removing each time a single feature dimension
        :param ux: initial user
        :return: a matrix of perturbed fake users
    """
    nb_dim = ux.size()[0]
    tmp = torch.tensor(ux[ux > 0])

    nb_non_zero_dim = tmp.size()[0]
    users = ux.expand(nb_non_zero_dim, nb_dim).clone()

    # for loop are bad in Python!
    row = 0
    col = 0
    for v in (ux > 0):
        if v:
            users[row][col] = 0
            row +=1
        col += 1


    return users

# Make Perturbations
def perturbations_4(ux, fake_users):

    """
    Generate random perturbed users from ux by exchanging 2 feature dimensions
    :param ux: initial user
    :param fake_users: number of perturbed instances based on user
    :return: a set of perturbed users
    """

    nb_dim = ux.size()[0]
    users = ux.expand(fake_users, nb_dim).clone()
    #print(users)

    tmp = torch.tensor(ux[ux > 0])
    # nb_non_zero_dim = tmp.size()[0]
    # print(nb_non_zero_dim)

    # small loop can't hurt
    r = range(nb_dim-1)
    non_zeros = np.argwhere((ux > 0).numpy())
    zeros = np.argwhere((ux == 0).numpy())

    for i in range(fake_users):
        a = non_zeros[np.random.randint(low=0,high=non_zeros.size)][0]
        b = zeros[np.random.randint(low=0,high=zeros.size)][0]
        v = users[i][a]
        users[i][a] = users[i][b]
        users[i][b] = v

    return users


def load_data_small():
    ratings_df = pd.read_csv("./ml-latest-small/ratings.csv")
    movies_df = pd.read_csv("./ml-latest-small/movies.csv")

    # Make it smaller
    ratings_df = ratings_df[ratings_df["userId"] < 51].drop(columns=['timestamp'])
    ratings_df = ratings_df.query("movieId < 400")
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


from scipy.spatial.distance import pdist, squareform


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


from math import sqrt, pow
from numpy.linalg import norm
def robustness(origin, origin_y, neighborhood, neighborhood_y):
    ratios = []
    for i, neighbor in enumerate(neighborhood):
        ratio = sqrt(pow(origin_y - neighborhood_y[i], 2))/(norm(origin - neighbor, 2))
        ratios.append(ratio)
    return max(ratios)


if __name__ == '__main__':
    U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data()
    mine = []

    print(np.nanstd(all_actual_ratings))

    for user in range(610):
        for e in range(1, 11):
            voisins = epsilon_neighborhood_fast(np.nan_to_num(all_actual_ratings), user, e/10.)
            if sum(voisins) > 0:
                mine.append(e)
                break

    print(np.mean(mine))