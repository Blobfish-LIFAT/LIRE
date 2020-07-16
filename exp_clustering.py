import torch
import numpy as np
from sklearn import linear_model
from models import LinearRecommender, train, get_OOS_pred, linear_recommender
from utility import load_data, perturbations, path_from_root, pick_cluster
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, to_tree
import loss
import random

# Gestion du mode pytorch CPU/GPU
from config import Config
random.seed(42)

Config.set_device_cpu()
device = Config.getInstance().device_
print("Running tensor computations on", device)

OUTFILE = "exp_clustering.csv"

with open(OUTFILE, mode="w") as file:
    file.write("CLS_MOVIE_WEIGHT,N_TRAINING_POINTS,PERTURBATION_RATIO,N_FEATS,PERT_STD,MAE" + '\n')




N_TRAINING_POINTS = 100
N_FEATS = 15
PERT_STD = 2
MOVIE_IDS = random.sample(range(2000), 25)
USER_IDS = random.sample(range(610), 10)

for pratio in [0.1, 0.5, 0.9]:
    for cls in [1, 2, 3, 4]:
        CLS_MOVIE_WEIGHT = cls
        PERTURBATION_RATIO = pratio
        PERTURBATION_NB = int(N_TRAINING_POINTS * PERTURBATION_RATIO)
        CLUSTER_NB = N_TRAINING_POINTS - PERTURBATION_NB

        U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data()

        for MOVIE_ID in MOVIE_IDS:
            print(" --- movie id", MOVIE_ID, "--- ")
            # build interp space with lasso
            other_movies = list(range(all_actual_ratings.shape[1]))
            other_movies.remove(MOVIE_ID)
            LX = np.nan_to_num(all_actual_ratings[:, other_movies])
            Ly = np.nan_to_num(all_user_predicted_ratings[:, MOVIE_ID])

            # LARS learning
            # complexity of the model is defined by N_FEATS: 30 features + intercept
            reg = linear_model.Lars(fit_intercept=True, n_nonzero_coefs=N_FEATS)
            reg.fit(LX, Ly)

            # select indexes of non 0 coefficients to determine the reduced space
            indexes = list(np.argwhere(reg.coef_ != 0).T.flatten())
            n_dim_int = np.sum(reg.coef_ != 0)

            # build distance matrix for clustering
            metric_indexes = indexes + [MOVIE_ID]
            w = np.ones(len(metric_indexes))
            w[len(indexes)] = float(CLS_MOVIE_WEIGHT)
            dist = pdist(np.nan_to_num(all_actual_ratings)[:, metric_indexes], 'wminkowski', p=2, w=w)

            linked = linkage(dist, 'ward')
            rootnode, nodelist = to_tree(linked, rd=True)

            for USER_ID in USER_IDS:
                path = path_from_root(rootnode, nodelist[USER_ID])
                try:
                    cluster_ids = pick_cluster(path)
                except ValueError:
                    continue
                print("user id", USER_ID, "cluster size", len(cluster_ids))
                try:
                    cluster_ids.remove(USER_ID)
                except ValueError:
                    pass
                try:
                    cluster_ids = random.sample(cluster_ids, CLUSTER_NB)
                except ValueError:
                    # Bypass ratio if not enough real points
                    cluster_ids = pick_cluster(path)
                    try:
                        cluster_ids.remove(USER_ID)
                    except ValueError:
                        pass
                    PERTURBATION_NB = N_TRAINING_POINTS - len(cluster_ids)

                # 1. Generate perturbations in interpretable space
                base_user = torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID, other_movies]), device=device,
                                         dtype=torch.float32)
                # here the interpretable space is a reduction of the initial space based on indexes
                base_user_int = base_user[reg.coef_ != 0]

                pert_int = perturbations(base_user_int, PERTURBATION_NB, std=PERT_STD)
                pert_orr = torch.zeros(PERTURBATION_NB, films_nb, device=device)

                # 2. generate perturbations in original space
                i = 0
                for pu in pert_int:
                    pert_orr[i] = torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID]), device=device, dtype=torch.float32)
                    j = 0
                    for index in indexes:
                        pert_orr[i][index] = pert_int[i][j]
                        j += 1
                    i += 1

                s = torch.tensor(sigma, device=device, dtype=torch.float32)
                v = torch.tensor(Vt, device=device, dtype=torch.float32)
                y_orr = get_OOS_pred(pert_orr, s, v, films_nb)

                # add points from cluster
                base_cluster = torch.tensor(np.nan_to_num(all_actual_ratings[:, other_movies])[cluster_ids], device=device,
                                            dtype=torch.float32)
                pert_int = torch.cat((pert_int, base_cluster[:, reg.coef_ != 0]))
                pert_orr = torch.cat((pert_orr, torch.tensor(np.nan_to_num(all_actual_ratings)[cluster_ids], device=device,
                                                             dtype=torch.float32)))
                y_orr = torch.cat((y_orr, torch.tensor(np.nan_to_num(all_user_predicted_ratings[cluster_ids]), device=device,
                                                       dtype=torch.float32)))

                models = []
                errors = []
                for i in range(10):
                    model = LinearRecommender(n_dim_int)

                    l = loss.LocalLossMAE_v3(
                        torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID]), device=device, dtype=torch.float32),
                        map_fn=lambda _: pert_orr, sigma=0.3)

                    train(model, pert_int, y_orr[:, MOVIE_ID], l, 100, verbose=False)

                    gx_ = model(base_user_int).item()
                    fx = all_user_predicted_ratings[USER_ID, MOVIE_ID]
                    errors.append(abs(gx_ - fx))
                    models.append(model)
                    if abs(gx_ - fx) < 0.1:
                        break

                best = errors[np.argmin(errors)]

                if best > 1.0:
                    print("Failover ! non zero dims ->", sum(base_user_int).item())
                    nonzero_idx = base_user != 0.
                    base_user_int = base_user[nonzero_idx]

                    # 1. Generate perturbations in interpretable space
                    pert_int = perturbations(base_user_int, PERTURBATION_NB, std=PERT_STD)
                    pert_orr = torch.zeros(PERTURBATION_NB, films_nb, device=device)

                    # 2. generate perturbations in original space
                    i = 0
                    for pu in pert_int:
                        pert_orr[i] = torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID]), device=device,
                                                   dtype=torch.float32)
                        j = 0
                        for index in indexes:
                            pert_orr[i][index] = pert_int[i][j]
                            j += 1
                        i += 1

                    y_orr = get_OOS_pred(pert_orr, s, v, films_nb)

                    # add points from cluster
                    base_cluster = torch.tensor(np.nan_to_num(all_actual_ratings[:, other_movies])[cluster_ids],
                                                device=device,
                                                dtype=torch.float32)
                    pert_int = torch.cat((pert_int, base_cluster[:, nonzero_idx]))
                    pert_orr = torch.cat((pert_orr, torch.tensor(np.nan_to_num(all_actual_ratings)[cluster_ids], device=device,
                                                dtype=torch.float32)))
                    y_orr = torch.cat(
                        (y_orr, torch.tensor(np.nan_to_num(all_user_predicted_ratings[cluster_ids]), device=device,
                                             dtype=torch.float32)))

                    alt = linear_model.Lars(fit_intercept=True, n_nonzero_coefs=N_FEATS)
                    alt.fit(pert_int, y_orr[:, MOVIE_ID])

                    #model = LinearRecommender(int(sum(nonzero_idx)))

                    #l = loss.LocalLossMAE_v3(
                    #    torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID]), device=device, dtype=torch.float32),
                    #    map_fn=lambda _: pert_orr, sigma=0.3)

                    #train(model, pert_int, y_orr[:, MOVIE_ID], l, 100, verbose=False)

                    gx_ = alt.predict(base_user_int.reshape(1, -1))
                    #gx_ = model(base_user_int).item()
                    fx = all_user_predicted_ratings[USER_ID, MOVIE_ID]
                    print('failover mae', abs(gx_ - fx))

                with open(OUTFILE, mode="a") as file:
                    out = [CLS_MOVIE_WEIGHT, N_TRAINING_POINTS, PERTURBATION_RATIO, N_FEATS, PERT_STD, best]
                    out = map(str, out)
                    file.write(','.join(out) + '\n')
