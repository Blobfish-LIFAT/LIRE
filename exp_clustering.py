import torch
import numpy as np
from sklearn import linear_model
from models import LinearRecommender, train, get_OOS_pred
from utility import load_data, get_user_cluster, k_neighborhood, robustness
from perturbations import perturbations_gaussian
from scipy.spatial.distance import pdist
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

OUTFILE = "res/exp_clustering_sigma.csv"
with open(OUTFILE, mode="w") as file:
    file.write("CLS_MOVIE_WEIGHT,N_TRAINING_POINTS,PERTURBATION_RATIO,sigma,PERT_STD,FAILOVER,MAE,MAE_FAILOVER,k,rob,rob_cold" + '\n')

N_TRAINING_POINTS = 50
PERT_STD = 1.04 #empiricaly determined from dataset
MOVIE_IDS = random.sample(range(2000), 10)
USER_IDS = random.sample(range(610), 10)
verbose = False

def gen_perturbations(pnb, base_user_int, all_actual_ratings, USER_ID, interp_indexes):
    """
    Generate perturbations and their image in original space
    :param pnb: number of perturbations
    :param base_user_int: user to perturbate in interpretable representation
    :param all_actual_ratings: original ratings matrix
    :param USER_ID: index of the user to perturbate
    :param interp_indexes: indexes of interpretable space
    :return: a pnb x |interp_indexes| matrix with the perturbations
    """
    pert_int = perturbations_gaussian(base_user_int, pnb, std=PERT_STD)
    pert_orr = torch.zeros(pnb, all_actual_ratings.shape[1], device=device)

    # 2. generate perturbations in original space
    for i, pu in enumerate(pert_int):
        pert_orr[i] = torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID]), device=device, dtype=torch.float32)
        j = 0
        for index in interp_indexes:
            pert_orr[i][index] = pert_int[i][j]
            j += 1

    return pert_int, pert_orr


def run_classic(all_actual_ratings_, all_user_predicted_ratings_, s_, v_, PERTURBATION_NB_, USER_ID_, MOVIE_ID_, interp_space_indexes_, cluster_ids_, LIRE_SIGMA=0.3):
    films_nb_ = all_actual_ratings_.shape[1]
    other_movies_ = list(range(all_actual_ratings_.shape[1]))
    other_movies_.remove(MOVIE_ID_)
    # 1. Generate perturbations in interpretable space
    # here the interpretable space is a reduction of the initial space based on indexes
    base_user_int = base_user[reg.coef_ != 0]
    pert_int, pert_orr = gen_perturbations(PERTURBATION_NB_, base_user_int, all_actual_ratings_, USER_ID_, interp_space_indexes_)
    y_orr = get_OOS_pred(pert_orr, s_, v_, films_nb_)

    # add points from cluster
    base_cluster = torch.tensor(np.nan_to_num(all_actual_ratings_[:, other_movies_])[cluster_ids_], device=device, dtype=torch.float32)
    pert_int = torch.cat((pert_int, base_cluster[:, reg.coef_ != 0]))
    pert_orr = torch.cat((pert_orr, torch.tensor(np.nan_to_num(all_actual_ratings_[cluster_ids_]), device=device,
                                                 dtype=torch.float32)))
    y_orr = torch.cat((y_orr, torch.tensor(np.nan_to_num(all_user_predicted_ratings_[cluster_ids_]), device=device,
                                           dtype=torch.float32)))

    models_ = []
    errors_ = []
    # A few runs to avoid bad starts
    for _ in range(10):
        model = LinearRecommender(len(interp_space_indexes_))
        local_loss = loss.LocalLossMAE_v3(
            torch.tensor(np.nan_to_num(all_actual_ratings_[USER_ID_]), device=device, dtype=torch.float32),
            map_fn=lambda _: pert_orr, sigma=LIRE_SIGMA)

        train(model, pert_int, y_orr[:, MOVIE_ID], local_loss, 100, verbose=False)

        gx = model(base_user_int).item()
        errors_.append(abs(gx - all_user_predicted_ratings_[USER_ID_, MOVIE_ID_]))
        models_.append(model)
        if abs(gx - all_user_predicted_ratings_[USER_ID_, MOVIE_ID_]) < 0.1:  # Good enough
            break

    return errors_, models_


def run_cold(all_actual_ratings_, all_user_predicted_ratings_, s_, v_, PERTURBATION_NB_, USER_ID_, N_FEATS_, other_movies_, cluster_ids_, base_user_, nonzero_idx):
    base_user_int = base_user_[nonzero_idx]

    pert_int, pert_orr = gen_perturbations(PERTURBATION_NB_, base_user_int, all_actual_ratings_, USER_ID_,
                                           np.argwhere(nonzero_idx).flatten())
    y_orr = get_OOS_pred(pert_orr, s_, v_, all_actual_ratings_.shape[1])

    # add points from cluster
    base_cluster = torch.tensor(np.nan_to_num(all_actual_ratings_[:, other_movies_])[cluster_ids_], device=device,
                                dtype=torch.float32)
    pert_int = torch.cat((pert_int, base_cluster[:, nonzero_idx]))

    y_orr = torch.cat((y_orr,
                       torch.tensor(np.nan_to_num(all_user_predicted_ratings_[cluster_ids_]), device=device,
                                    dtype=torch.float32)))

    alt = linear_model.Lars(fit_intercept=True, n_nonzero_coefs=N_FEATS_)
    alt.fit(pert_int, y_orr[:, MOVIE_ID])

    return alt, alt.predict(base_user_int.reshape(1, -1))


N_FEATS = 10
for lire_sigma in [1,2,4]:
    for pratio in [0., 0.5, 1.0]:
        for cls in [10]:
            CLS_MOVIE_WEIGHT = cls
            PERTURBATION_RATIO = pratio
            PERTURBATION_NB = int(N_TRAINING_POINTS * PERTURBATION_RATIO)
            CLUSTER_NB = N_TRAINING_POINTS - PERTURBATION_NB

            U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data()
            s = torch.tensor(sigma, device=device, dtype=torch.float32)
            v = torch.tensor(Vt, device=device, dtype=torch.float32)

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
                interp_space_indexes = list(np.argwhere(reg.coef_ != 0).T.flatten())

                # Clustering - Metric and linkage matrix
                metric_indexes = interp_space_indexes + [MOVIE_ID]
                w = np.ones(len(metric_indexes))
                w[len(interp_space_indexes)] = float(CLS_MOVIE_WEIGHT)
                points = np.nan_to_num(all_actual_ratings)
                points[:, MOVIE_ID] = all_user_predicted_ratings[:, MOVIE_ID]
                dist = np.clip(pdist(points, cosine), 0., np.inf)
                #dist = pdist(points[:, metric_indexes], 'wminkowski', p=2, w=w)

                linked = linkage(dist, 'ward')
                rootnode, nodelist = to_tree(linked, rd=True)

                for USER_ID in USER_IDS:
                    base_user = torch.tensor(np.nan_to_num(all_actual_ratings[USER_ID, other_movies]), device=device,
                                             dtype=torch.float32)
                    fx = all_user_predicted_ratings[USER_ID, MOVIE_ID]

                    # Find out user's cluster
                    cluster_ids = get_user_cluster(USER_ID, rootnode, nodelist[USER_ID])
                    if cluster_ids is None:
                        continue
                    if len(cluster_ids) > CLUSTER_NB:
                        cluster_ids = random.sample(cluster_ids, CLUSTER_NB)
                    else:
                        print("[WARNING] Not Enough points")

                    errors, models = run_classic(all_actual_ratings, all_user_predicted_ratings, s, v, PERTURBATION_NB, USER_ID, MOVIE_ID, interp_space_indexes, cluster_ids, LIRE_SIGMA=lire_sigma)
                    classic_mae = errors[np.argmin(errors)]

                    alt_model, gx_ = run_cold(all_actual_ratings, all_user_predicted_ratings, s, v, PERTURBATION_NB, USER_ID, N_FEATS, other_movies, cluster_ids, base_user, base_user != 0.)
                    cold_mae = abs(gx_ - fx)

                    # --- Robustness stuff ---
                    k_indexes = k_neighborhood(np.nan_to_num(all_actual_ratings), USER_ID, 15)

                    # Robustness for classic
                    k_points = np.nan_to_num(all_actual_ratings[k_indexes])[:,
                               other_movies]  # TODO maybe on all movies ?
                    k_omega = torch.zeros(15, len(interp_space_indexes))

                    for i, neighbor in enumerate(k_indexes):
                        neighbor_cluster_ids = get_user_cluster(neighbor, rootnode, nodelist[neighbor])
                        if neighbor_cluster_ids is None:
                            continue
                        if len(neighbor_cluster_ids) > CLUSTER_NB:
                            neighbor_cluster_ids = random.sample(neighbor_cluster_ids, CLUSTER_NB)

                        neighbor_errors, neighbor_models = run_classic(all_actual_ratings, all_user_predicted_ratings,
                                                                       s, v, PERTURBATION_NB, neighbor, MOVIE_ID,
                                                                       interp_space_indexes, neighbor_cluster_ids, LIRE_SIGMA=lire_sigma)
                        k_omega[i] = neighbor_models[np.argmin(neighbor_errors)].omega

                    # Robustness for cold
                    """k_coefs = np.zeros((15, alt_model.coef_.shape[0]))

                    for i, neighbor in enumerate(k_indexes):
                        neighbor_cluster_ids = get_user_cluster(neighbor, rootnode, nodelist[neighbor])
                        if neighbor_cluster_ids is None:
                            continue
                        if len(neighbor_cluster_ids) > CLUSTER_NB:
                            neighbor_cluster_ids = random.sample(neighbor_cluster_ids, CLUSTER_NB)

                        neighbor_alt_model, neighbor_gx_ = run_cold(all_actual_ratings, all_user_predicted_ratings,
                                                                    s, v, PERTURBATION_NB, neighbor, N_FEATS,
                                                                    other_movies, neighbor_cluster_ids,
                                                                    torch.tensor(np.nan_to_num(
                                                                        all_actual_ratings[neighbor, other_movies]),
                                                                                 device=device,
                                                                                 dtype=torch.float32),
                                                                    base_user != 0.)
                        k_coefs[i] = neighbor_alt_model.coef_"""

                    for k in [5, 10, 15]:
                        rob_classic = robustness(base_user.detach().numpy(), models[np.argmin(errors)].omega.detach().numpy(), k_points[:k+1], k_omega.detach().numpy()[:k+1])# Robustness for classic
                        #rob_cold = robustness(base_user.detach().numpy(), alt_model.coef_, k_points[:k+1], k_coefs[:k+1])
                        rob_cold = 0

                        out = [CLS_MOVIE_WEIGHT, N_TRAINING_POINTS, PERTURBATION_RATIO, lire_sigma, PERT_STD, sum(base_user[reg.coef_ != 0]).item() == 0, classic_mae, cold_mae[0], k, rob_classic, rob_cold]
                        out = map(str, out)
                        out_lines.append(','.join(out) + '\n')

                with open(OUTFILE, mode="a+") as file:
                    file.writelines(out_lines)
                    out_lines.clear()


"""
pert_int = perturbations_3(base_user_int)
if PERTURBATION_NB == 0:
    pert_orr = torch.zeros(0, films_nb, device=device)
    pert_int = torch.zeros(0, sum(reg.coef_ != 0))
else:
    pert_orr = torch.zeros(pert_int.size()[0], films_nb, device=device)
"""