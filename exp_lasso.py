import torch
import numpy as np
from sklearn import linear_model
from utility import load_data_small, perturbations
import seaborn as sns
import matplotlib.pyplot as plt
from models import LinearRecommender, get_OOS_pred_inner, train
import loss

# Gestion du mode pytorch CPU/GPU
from config import Config

device = Config.getInstance().device_
print("Running tensor computations on", device)

# Load data and run black box
U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data_small()

# produce an inner interpretable space with LARS


# -------------
movie_id = 69  # 55 => pb with Lars model, 69 seems to work just fine
n_tests = 25
n_neighbors = 25
N_FEATS = 10
# -------------

# here the best reduced feature space is learned based on all users
# !!! This should be changed in the end so as to keep training / testing more fair !!!
other_movies = list(range(all_actual_ratings.shape[1]))
other_movies.remove(movie_id)
LX = np.nan_to_num(all_actual_ratings[:, other_movies])
Ly = np.nan_to_num(all_user_predicted_ratings[:, movie_id])

# LARS learning
# complexity of the model is defined by N_FEATS: 30 features + intercept

reg = linear_model.Lars(fit_intercept=True, n_nonzero_coefs=N_FEATS)
reg.fit(LX, Ly)

# select indexes of non 0 coefficients to determine the reduced space
indexes = np.argwhere(reg.coef_ != 0).T.flatten()

# -------------
n_dim_int = np.sum(reg.coef_ != 0)
print(n_dim_int)
# -------------

# from here - COPY/PASTE from Alex'Code (in Enchanted World)


s = torch.tensor(sigma, device=device, dtype=torch.float32)
v = torch.tensor(Vt, device=device, dtype=torch.float32)

# --------------

# 1. Generate perturbations in interpretable space

for user_id in range(n_tests):
    base_user = torch.tensor(np.nan_to_num(all_actual_ratings[user_id, other_movies]), device=device,
                             dtype=torch.float32)
    # here the interpretable space is a reduction of the initial space based on indexes
    base_user_int = base_user[reg.coef_ != 0]
    # if user_id == 0:
    #    print("first values of base_user:", base_user[indexes.T[0,0:6]])
    #    print("first line base_user_int", base_user_int[0:6])
    pert_int = perturbations(base_user_int, n_neighbors, )
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

vprime = v[:, other_movies]
y_orr = get_OOS_pred_inner(pert_orr, s, vprime, films_nb)

errors = []
for movie_id in range(50, 150):
    model = LinearRecommender(n_dim_int)


    def map_fn(entree):
        return pert_orr


    l = loss.LocalLossMAE_v3(base_user, movie_id, map_fn=map_fn)
    train(model, pert_int, y_orr[:, movie_id], l, 100, verbose=False)

    gx_ = model(base_user_int).item()
    fx = y_orr[:, movie_id].mean().item()
    # print(gx_, fx, all_user_predicted_ratings[user_id, movie_id])
    errors.append(abs(gx_ - fx))
    # print(abs(gx_ - fx))

errors = np.array(errors)
print(errors.mean(), errors.std())
sns.distplot(errors)
plt.show()
print("done")
