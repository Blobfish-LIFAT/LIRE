import torch
import numpy as np
from numpy import savetxt
from sklearn import linear_model
from models import LinearRecommender, train, get_OOS_pred, linear_recommender
from utility import load_data, perturbations_gaussian, path_from_root, pick_cluster, perturbations_3
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import cosine
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
    file.write("CLS_MOVIE_WEIGHT,N_TRAINING_POINTS,PERTURBATION_RATIO,N_FEATS,PERT_STD,FAILOVER,MAE,MAE_FAILOVER,rob,rob_cold" + '\n')

N_TRAINING_POINTS = 50
N_FEATS = 15
PERT_STD = 1.04 #empiricaly determined from dataset
MOVIE_IDS = random.sample(range(2000), 25)
USER_IDS = random.sample(range(610), 15)
verbose = False



for N_FEATS in [10]:
    for pratio in [0.0, 0.5, 1.0]:
        for cls in [4]:
            CLS_MOVIE_WEIGHT = cls
            PERTURBATION_RATIO = pratio
            PERTURBATION_NB = int(N_TRAINING_POINTS * PERTURBATION_RATIO)
            CLUSTER_NB = N_TRAINING_POINTS - PERTURBATION_NB
            U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data()

            out_lines = []
            for MOVIE_ID in MOVIE_IDS:
                # Build interp space with lasso
                # complexity of the model is defined by N_FEATS
                other_movies = list(range(all_actual_ratings.shape[1]))
                other_movies.remove(MOVIE_ID)
                LX = np.nan_to_num(all_actual_ratings[:, other_movies])
                Ly = np.nan_to_num(all_user_predicted_ratings[:, MOVIE_ID])

                # LARS learning
                reg = linear_model.Lars(fit_intercept=True, n_nonzero_coefs=N_FEATS)
                reg.fit(LX, Ly)

                # select indexes of non 0 coefficients to determine the reduced space
                indexes = list(np.argwhere(reg.coef_ != 0).T.flatten())
                n_dim_int = np.sum(reg.coef_ != 0)

                # Clustering
                metric_indexes = indexes + [MOVIE_ID]
                w = np.ones(len(metric_indexes))
                w[len(indexes)] = float(CLS_MOVIE_WEIGHT)
                points = np.nan_to_num(all_actual_ratings)
                points[:, MOVIE_ID] = all_user_predicted_ratings[:, MOVIE_ID]
                dist = np.clip(pdist(points, cosine), 0., np.inf)
                #dist = pdist(points[:, metric_indexes], 'wminkowski', p=2, w=w)

                linked = linkage(dist, 'ward')
                rootnode, nodelist = to_tree(linked, rd=True)

                for USER_ID in USER_IDS:
                    path = path_from_root(rootnode, nodelist[USER_ID])

                    try:
                        cluster_ids = pick_cluster(path)
                    except ValueError:
                        print("[ERROR] Error on clustering")
                        continue

                    if USER_ID in cluster_ids:
                        cluster_ids.remove(USER_ID)

                    if verbose: print("movie id", MOVIE_ID, "user id", USER_ID, "cluster size", len(cluster_ids))

                    try:
                        cluster_ids = random.sample(cluster_ids, CLUSTER_NB)
                    except ValueError:
                        print("[WARNING] Not Enough points")


                    # 1. Generate perturbations in interpretable space
                    base_user = torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID, other_movies]), device=device,
                                             dtype=torch.float32)
                    # here the interpretable space is a reduction of the initial space based on indexes
                    base_user_int = base_user[reg.coef_ != 0]

                    pert_int = perturbations_gaussian(base_user_int, PERTURBATION_NB)
                    pert_orr = torch.zeros(pert_int.size()[0], films_nb, device=device)

                    # 2. generate perturbations in original space
                    for i, pu in enumerate(pert_int):
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
                    pert_orr = torch.cat((pert_orr, torch.tensor(np.nan_to_num(all_actual_ratings[cluster_ids]), device=device,
                                                                 dtype=torch.float32)))
                    y_orr = torch.cat((y_orr, torch.tensor(np.nan_to_num(all_user_predicted_ratings[cluster_ids]), device=device,
                                                           dtype=torch.float32)))

                    models = []
                    errors = []
                    # A few runs to avoid bad starts
                    for i in range(10):
                        model = LinearRecommender(n_dim_int)
                        l = loss.LocalLossMAE_v3(torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID]), device=device, dtype=torch.float32), map_fn=lambda _: pert_orr, sigma=0.3)

                        train(model, pert_int, y_orr[:, MOVIE_ID], l, 100, verbose=False)

                        gx_ = model(base_user_int).item()
                        fx = all_user_predicted_ratings[USER_ID, MOVIE_ID]
                        errors.append(abs(gx_ - fx))
                        models.append(model)
                        if abs(gx_ - fx) < 0.1:# Good enough
                            break

                    best = errors[np.argmin(errors)]

                    if best > 0.0:
                        if verbose: print("Failover ! non zero dims ->", sum(base_user_int).item())
                        fail_enable = True
                        nonzero_idx = base_user != 0.
                        base_user_int = base_user[nonzero_idx]

                        # 1. Generate perturbations in interpretable space
                        pert_int = perturbations_gaussian(base_user_int, PERTURBATION_NB, std=PERT_STD)
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
                        pert_orr = torch.cat((pert_orr, torch.tensor(np.nan_to_num(all_actual_ratings[cluster_ids]), device=device,
                                                    dtype=torch.float32)))
                        y_orr = torch.cat(
                            (y_orr, torch.tensor(np.nan_to_num(all_user_predicted_ratings[cluster_ids]), device=device,
                                                 dtype=torch.float32)))

                        alt = linear_model.Lars(fit_intercept=True, n_nonzero_coefs=N_FEATS)
                        alt.fit(pert_int, y_orr[:, MOVIE_ID])

                        gx_ = alt.predict(base_user_int.reshape(1, -1))
                        fx = all_user_predicted_ratings[USER_ID, MOVIE_ID]
                        best_fail = abs(gx_ - fx)

                        if verbose: print('failover mae', best_fail, "previous", best)

                        best_model = alt
                    else:
                        fail_enable = False
                        best_model = models[np.argmin(errors)]

                    if verbose: print('--- Robustness testing ---')
                    from utility import robustness, epsilon_neighborhood_fast

                    uvec = np.nan_to_num(all_actual_ratings[USER_ID])
                    uy = best_model.predict(base_user_int.reshape(1, -1))

                    nids = epsilon_neighborhood_fast(np.nan_to_num(all_actual_ratings), USER_ID, 0.7).flatten()
                    if len(nids) > 0:
                        npoints = np.nan_to_num(all_actual_ratings[nids])
                        try:
                            rob_cold = robustness(uvec, uy, npoints, best_model.predict(npoints[:, torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID, :]), device=device, dtype=torch.float32) != 0 if fail_enable else reg.coef_ != 0]))
                        except ValueError:
                            rob_cold = np.nan
                        rob = robustness(uvec, models[np.argmin(errors)](base_user[reg.coef_ != 0]).detach().numpy(), npoints, models[np.argmin(errors)].predict(npoints[:, other_movies][:, reg.coef_ != 0]))
                    else:
                        rob = rob_cold = np.nan

                    out = [CLS_MOVIE_WEIGHT, N_TRAINING_POINTS, PERTURBATION_RATIO, N_FEATS, PERT_STD, best > 2, best, best_fail[0], rob, rob_cold]
                    out = map(str, out)
                    out_lines.append(','.join(out) + '\n')

                with open(OUTFILE, mode="a+") as file:
                    file.writelines(out_lines)


"""
pert_int = perturbations_3(base_user_int)
if PERTURBATION_NB == 0:
    pert_orr = torch.zeros(0, films_nb, device=device)
    pert_int = torch.zeros(0, sum(reg.coef_ != 0))
else:
    pert_orr = torch.zeros(pert_int.size()[0], films_nb, device=device)
"""