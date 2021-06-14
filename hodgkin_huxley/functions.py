# functions for all methods

import inspect
import os
import sys

from torch import nn
import torch
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.distributions.uniform import BoxUniform

from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import (
    CompositeTransform,
)
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.standard import PointwiseAffineTransform
from nflows.transforms.nonlinearities import Tanh
from nflows.transforms.normalization import BatchNorm
import numpy as np

# load from util (from https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from util import InvSigmoid
from util import UniformContext

# Return min ess of the samples in x, code adapted from https://github.com/gpapamak/snl/blob/master/util/math.py
def ess_mcmc(x):

    N, dim = x.shape

    x = x - x.mean(0)

    acors = torch.zeros(x.shape)

    for i in range(dim):
        for lag in range(N):
            acor = torch.dot(x[:N-lag, i], x[lag:, i]) / (N - lag)
            if acor < 0:
                break
            acors[lag, i] = acor

    act = 1 + 2 + acors[1:].sum(0) / acors[0]
    ess = N/act

    return min(ess).item()

def load_summary_stats_mean_and_std():
    m_s_of_prior = np.loadtxt('data/m_s_of_prior.csv', delimiter=",")
    s_s_of_prior = np.loadtxt('data/s_s_of_prior.csv', delimiter=",")

    return torch.from_numpy(m_s_of_prior).to(dtype=torch.float32), \
           torch.from_numpy(s_s_of_prior).to(dtype=torch.float32)


# For whiten, calc_whitening_transform, and de_whiten: code adapted from: https://github.com/gpapamak/snl/blob/master/util/math.py
def whiten(xs, params):
    """
    Whitens a given dataset using the whitening transform provided.
    """

    means, U, istds = params

    ys = xs.copy()
    ys -= means
    ys = np.dot(ys, U)
    ys *= istds

    return ys


def calc_whitening_transform(xs):
    """
    Calculates the parameters that whiten a dataset.
    """

    N = xs.shape[0]

    means = np.mean(xs, axis=0)
    ys = xs - means

    cov = np.dot(ys.T, ys) / N
    vars, U = np.linalg.eig(cov)
    istds = np.sqrt(1.0 / vars)

    return means, U, istds

def de_whiten(xs, params):
    """
    De-whitens a given dataset using the whitening transform provided.
    """

    means, U, istds = params

    ys = xs.copy()
    ys /= istds
    ys = np.dot(ys, U.T)
    ys += means

    return ys

def pilot_run(model, simulator, summary_stats_obs, plotting=True, nbr_prior_samples=5000):

    # local imports
    from tqdm import tqdm

    n_summary = 19
    prior_samples = model.prior.sample((nbr_prior_samples,))

    data_from_prior = np.zeros((nbr_prior_samples, n_summary))

    torch.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for i in tqdm(range(nbr_prior_samples), desc="Pilot run: running " + str(nbr_prior_samples) + " simulations"):
        data_from_prior[i, :] = simulator(prior_samples[i, :])

    means, U, istds = calc_whitening_transform(data_from_prior)

    np.savetxt('data/whitness_means.csv', means, delimiter=",")
    np.savetxt('data/whitness_U.csv', U, delimiter=",")
    np.savetxt('data/whitness_istds.csv', istds, delimiter=",")


    return means, U, istds


# sets up the networks for the flow and likelihood and posterior model
def set_up_networks(lower_post_limits, upper_post_limits, dim_post=12, dim_summary_stat=19, seed=10):
    torch.manual_seed(seed)
    base_dist_lik = StandardNormal(shape=[dim_summary_stat])

    num_layers = 4

    transforms = []
    for _ in range(num_layers):  # TODO add inv sigmoide fnunc
        transforms.append(ReversePermutation(features=dim_summary_stat))
        transforms.append(MaskedAffineAutoregressiveTransform(features=dim_summary_stat,
                                                              hidden_features=40,
                                                              context_features=dim_post,
                                                              num_blocks=2))

    transform = CompositeTransform(transforms)

    flow_lik = Flow(transform, base_dist_lik)

    base_dist_post = StandardNormal(shape=[dim_post])
    #base_dist_post = ConditionalDiagonalNormal(shape=[dim_post], context_encoder=nn.Linear(dim_summary_stat, 2*dim_post))

    #base_dist_post = UniformContext.UniformContext(low=lower_post_limits, high=upper_post_limits, shape=[dim_post])

    num_layers = 4

    transforms = []

    # def post model in prior snplace
    shift_vec = torch.zeros(dim_post)
    scale_vec = torch.zeros(dim_post)
    num_off_set = 0.001  # numerical offset since the prior is on the open space

    print("set priors")

    print(upper_post_limits)
    print(lower_post_limits)

    for i in range(dim_post):

        shift_tmp, scale_tmp = calc_scale_and_shift(lower_post_limits[i], upper_post_limits[i])
        shift_vec[i] = shift_tmp
        scale_vec[i] = scale_tmp

    #print(shift_vec)
    #print(scale_vec)

    print(shift_vec)
    print(scale_vec)


    # last transformation
    transforms.append(PointwiseAffineTransform(shift=shift_vec, scale=scale_vec))
    transforms.append(InvSigmoid.InvSigmoid())  # this should be inv sigmoide!

    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=dim_post))
        transforms.append(MaskedAffineAutoregressiveTransform(features=dim_post,
                                                              hidden_features=50,
                                                              context_features=dim_summary_stat,
                                                              num_blocks=2))

    transform = CompositeTransform(transforms)

    flow_post = Flow(transform, base_dist_post)

    return flow_lik, flow_post


def calc_scale_and_shift(lower, upper):
    sigma_lower = 0
    sigma_upper = 1

    scale = (sigma_upper - sigma_lower) / (upper - lower)
    shift = sigma_lower - lower * scale

    return shift, scale


