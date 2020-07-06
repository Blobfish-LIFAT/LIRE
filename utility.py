import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix

from config import Config

# Make Perturbations
def perturbations(ux, fake_users, std=2, proba=0.1):
    nb_dim = ux.size()[0]
    users = ux.expand(fake_users, nb_dim)

    perturbation = nn.init.normal_(torch.zeros(fake_users, nb_dim, device=Config.device()), 0, std)
    rd_mask = torch.zeros(fake_users, nb_dim, device=Config.device()).uniform_() > (1. - proba)
    perturbation = perturbation * rd_mask * (users != 0.)
    users = users + perturbation
    users[users > 5.] = 5.
    return torch.abs(users)


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