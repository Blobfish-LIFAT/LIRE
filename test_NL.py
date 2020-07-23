import torch
from perturbations import perturbations_swap, perturbations_gaussian
import numpy as np
from math import sqrt, pow
from numpy.linalg import norm
from utility import load_data
import random as rd
from scipy.spatial.distance import cosine

"""
user = torch.tensor([1, 0, 4, 0, 0, 0, 2])
res = perturbations_3(user)
res = perturbations_swap(user, 10)
print(res)
"""

# test cosine distance
def test_cosine_dist(nb_tests):
    dist = np.zeros((nb_tests))
    for i in range(nb_tests):
        # setting a number of dimensions
        dim = rd.randint(1, 10000)
        # generating vectors
        u = np.random.randint(-5, 5, dim)
        v = np.random.randint(-5, 5, dim)

        dist[i] = cosine(u, v)
        if dist[i] < 0:
            print("We might have a problem (d < 0)")
        elif dist[i] > 1:
            print("I spotted a distance greater than 1! (but that's expected)")
        elif dist[i] > 2:
            print("We might have a problem (d > 2)")
    print("done")

test_cosine_dist(1000)

"""
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
def robustness(origin, origin_y, neighborhood, neighborhood_y):
    ratios = []
    for i, neighbor in enumerate(neighborhood):
        ratio = cosine_dist(origin_y, neighborhood_y[i]) / cosine_dist(origin, neighbor)
        ratios.append(ratio)
    if max(ratios) == 0.:
        print("debug")
    return max(ratios)



U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data()
MOVIE_ID = 500

for user in range(610):
    voisins = k_neighborhood(np.nan_to_num(all_actual_ratings), user, 10)
    rob = robustness(np.nan_to_num(all_actual_ratings[user]), all_user_predicted_ratings[user, MOVIE_ID], np.nan_to_num(all_actual_ratings[voisins]), all_user_predicted_ratings[voisins, MOVIE_ID])
    print(" ", "Robustness", rob)

"""



"""
for user in range(610):
    print(k_neighborhood(np.nan_to_num(all_actual_ratings), user, 10))
    for e in range(1, 11):
        voisins = epsilon_neighborhood_fast(np.nan_to_num(all_actual_ratings), user, e/10.)
        if sum(voisins) > 0:
            print("--- epsilon", e/10., "for", user)
            rob = robustness(np.nan_to_num(all_actual_ratings[user]), all_user_predicted_ratings[user, MOVIE_ID], np.nan_to_num(all_actual_ratings[voisins]), all_user_predicted_ratings[voisins, MOVIE_ID])
            print(" ", "Robustness", rob)
            break
"""
