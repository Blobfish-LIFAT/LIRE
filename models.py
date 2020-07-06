import torch
import torch.nn as nn
import torch.optim as optim

from config import Config


class LinearRecommender(nn.Module):
    def __init__(self, in_shape, decay=0.3):
        super(LinearRecommender, self).__init__()
        self.dims = in_shape  # number of dimensions (eg movies)
        self.decay = torch.tensor(decay, device=Config.device())
        self.omega = nn.Parameter(
            torch.abs(nn.init.normal_(torch.zeros((self.dims,), device=Config.device()), mean=.0, std=0.1)))

    def forward(self, x):
        # init output
        if len(x.size()) > 1:
            y = (torch.sum(x * self.omega.expand(x.size()[0], self.dims), axis=1) / torch.sum(self.omega))
        else:
            y = (torch.sum(x * self.omega) / torch.sum(self.omega))
        # for each user
        # for u in range(x.size()[0]):
        # compute sim to all other users
        # sim = torch.sum(x[u]*x, 1) / ( torch.sqrt(torch.sum((x[u])**2)) * torch.sqrt(torch.sum((x)**2, 1)) )
        # sim = torch.exp(-0.5*torch.pow((sim-1)/self.decay ,2))
        # norm = torch.sum(sim)

        # y[u] =  torch.sum( (torch.sum(x * self.omega.expand(x.size()[0], self.dims), axis = 1) /  torch.sum(self.omega)) * sim ) / norm

        return y


# Training loop
# target_user_vec -> user targeted for explaination (as a 'real' ratings vector)
def train(model, X, y, loss_criterion, epochs, learning_rate=0.1, verbose=True):
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):  # loop over the dataset multiple times
        if verbose:
            print("epoch", epoch, end='')
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(X)
        loss = loss_criterion(X, model.omega, outputs, y)
        loss.backward()
        optimizer.step()

        # Clamp wheights to 0 no less
        # model.omega.data[model.omega < 0.] = 0.

        # print statistics
        if verbose:
            print(" | loss", loss.item())


# Quick Out of sample prediction for matrix factorization, we try to get the user's vector in the latent space
# that minimizes the error reconstructing it's ratings
# TODO -> rewrite for several users and do more test
def get_OOS_pred(user, s, v, films_nb, epochs=20):
    # print("  --- --- ---")
    umean = user.sum(axis=1) / (user != 0.).sum(axis=1)
    umask = user != 0.

    unew = nn.Parameter(torch.zeros(user.size()[0], s.size()[0], device=user.device, requires_grad=True))
    opt = optim.Adagrad([unew])

    for epoch in range(epochs):
        pred = unew @ s @ v + umean.expand(films_nb, user.size()[0]).transpose(0, 1)
        loss = torch.sum(torch.pow(((user - pred) * umask), 2)) / torch.sum(umask)
        loss.backward()
        opt.step()
        opt.zero_grad()
        # unew.data -= 0.0007 * unew.grad.data
        # unew.grad.data.zero_()
        # if epoch == 0 or epoch % 2 == 0 : print("  ", loss.item())

    return ((unew @ s @ v + umean.expand(films_nb, user.size()[0]).transpose(0, 1)) * (user == 0.) + user).detach()


# Quick Out of sample prediction for matrix factorization
# WITH REDUCED SET OF DIMENSIONS
# Minimizes the error reconstructing it's ratings
# TODO -> same as get_OOS_pred
def get_OOS_pred_inner(user, s, v, films_nb, epochs=20):
    # print("  --- --- ---")
    umean = user.sum(axis=1) / (user != 0.).sum(axis=1)
    umask = user != 0.

    unew = nn.Parameter(torch.zeros(user.size()[0], s.size()[0], device=user.device, requires_grad=True))
    opt = optim.Adagrad([unew])

    for epoch in range(epochs):
        # Warning: nb_films-1 as we work in a space where a movie was removed to be explained
        pred = unew @ s @ v + umean.expand(films_nb - 1, user.size()[0]).transpose(0, 1)
        loss = torch.sum(torch.pow(((user - pred) * umask), 2)) / torch.sum(umask)
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Warning: nb_films-1 as we work in a space where a movie was removed to be explained
    return ((unew @ s @ v + umean.expand(films_nb - 1, user.size()[0]).transpose(0, 1)) * (user == 0.) + user).detach()