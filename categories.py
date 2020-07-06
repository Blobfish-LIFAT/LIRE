import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import Config


class GenresSpace:
    def __init__(self, movies_df, ratings_df, all_actual_ratings):
        # !!! Be Careful !!! user and movies ID off by one in the DF (starts at 1 instead of 0 in matrices)
        self.g_names = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                        'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.all_actual_ratings = all_actual_ratings

        # Make a genre x movies binary mask
        self.genres_mask = np.array([movies_df['genres'].str.contains(g) for g in self.g_names])
        self.genres_mask = self.genres_mask.transpose()[list(set(ratings_df['movieId'] - 1))]

        self.G = torch.tensor(self.genres_mask.transpose(), device=Config.device(), dtype=torch.float32)
        #   G (genre, movies)           Rt (movies, userss)
        self.A = self.G @ torch.tensor(np.nan_to_num(all_actual_ratings.transpose()), device=Config.device(), dtype=torch.float32)

    def to_int_space(self, users):
        return self.G @ users

    # User_int is the perturbation to send back to original space, base_user_int is the 'real' user for reference
    def to_orr_space(self, user_int, base_user_int, base_user, verbose=False):
        dummy_user_orr = torch.tensor(np.nanmean(self.all_actual_ratings, axis=0), device=Config.device(), dtype=torch.float32)
        params = []
        changed_cat = (base_user_int - user_int) != 0.

        m_index = 0
        for movie in dummy_user_orr:
            check = False
            c_index = 0
            for mcat in changed_cat:
                check = check or (mcat and self.G.transpose(0, 1)[m_index, c_index])
                c_index += 1
            if check:
                params.append(True)
            else:
                dummy_user_orr[m_index] = base_user[m_index].clone().detach()
                params.append(False)
            m_index += 1

        params = torch.tensor(params, device=Config.device())
        dummy_user_orr = dummy_user_orr.clone().detach().requires_grad_(True)
        opt = optim.Adagrad([dummy_user_orr], lr=0.4)

        for epoch in range(300):
            opt.zero_grad()

            estimation = self.G @ dummy_user_orr
            loss = torch.mean(torch.pow(user_int - estimation, 2))

            loss.backward()
            # zero gradient on fixed ratings
            dummy_user_orr.grad *= params
            opt.step()

            dummy_user_orr.data[dummy_user_orr < 0.] = 0.

            if verbose and epoch % 10 == 0: print("epoch", epoch, "loss", loss)

        return dummy_user_orr.detach()
