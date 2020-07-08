import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from models import LinearRecommender, get_OOS_pred, train
from utility import load_data_small, perturbations
from categories import GenresSpace
import loss


# Gestion du mode pytorch CPU/GPU
from config import Config
Config.set_device_gpu()
device = Config.device()
print("Running tensor computations on", device)

configs = []
for n_feats in [5, 10, 15, 20]:
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for pert_std in [1, 2, 3, 4, 5]:
            configs.append((n_feats, sigma, pert_std))

# Load data and run black box
U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data_small()
int_space = GenresSpace(movies_df, ratings_df, all_actual_ratings)

errors = []
s = torch.tensor(sigma, device=device, dtype=torch.float32)
v = torch.tensor(Vt, device=device, dtype=torch.float32)

for user_id in range(10, 20):
    base_user = torch.tensor(np.nan_to_num(all_actual_ratings[user_id]), device=device, dtype=torch.float32)
    base_user_int = int_space.to_int_space(base_user)

    pert_int = perturbations(base_user_int, 25)
    pert_orr = torch.zeros(25, films_nb, device=device)

    i = 0
    for pu in pert_int:
        pert_orr[i] = int_space.to_orr_space(pert_int[i], base_user_int, base_user)
        i += 1

    y_orr = get_OOS_pred(pert_orr, s, v, films_nb)

    for movie_id in range(75, 100):
        model = LinearRecommender(18)
        l = loss.LocalLossMAE_v3(base_user, map_fn=lambda _: pert_orr, alpha=0.001)
        train(model, pert_int, y_orr[:, movie_id], l, 100, verbose=False, clamp=True)

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
