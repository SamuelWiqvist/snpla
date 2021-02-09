# functions for all methods

# load from util (from https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
import inspect
import os
import sys
from scipy import stats

import LotkaVolterra
import numpy as np
import torch
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import (
    CompositeTransform,
)
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.standard import PointwiseAffineTransform
from sbi.utils import BoxUniform
from torch.distributions.multivariate_normal import MultivariateNormal

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from util import InvSigmoid


def fit_gaussian_dist(post_samples):
    post_samples_np = post_samples.numpy()

    mean = np.mean(post_samples_np, axis=0)
    cov = np.cov(post_samples_np, rowvar=0)

    return MultivariateNormal(loc=torch.from_numpy(mean).to(dtype=torch.float32),
                              covariance_matrix=torch.from_numpy(cov).to(dtype=torch.float32))


def normalize_summary_stats(s, m_s, s_s):
    n = s.shape[0]

    for i in range(n):
        s[i, :] = (s[i, :] - m_s) / s_s

    return s


def load_summary_stats_mean_and_std():
    m_s_of_prior = np.loadtxt('data/m_s_of_prior.csv', delimiter=",")
    s_s_of_prior = np.loadtxt('data/s_s_of_prior.csv', delimiter=",")

    return torch.from_numpy(m_s_of_prior).to(dtype=torch.float32), \
           torch.from_numpy(s_s_of_prior).to(dtype=torch.float32)


def trim_std(s, cutoff=0.1):
    [lower, upper] = np.quantile(s, [cutoff, 1 - cutoff])
    return stats.tstd(s, limits=(lower, upper))


def gen_summary_stats_mean_and_std(model, nbr_sim=1000, seed=100, cutoff=0.0125, save=True):
    """
    Gen mean and std of the symmary stats for normalization

    :param save:
    :param cutoff:
    :param model: LV model
    :param nbr_sim: nbr of sims for comp mean and std
    :param seed: random seed
    :return: mean(s_of_paths) tensor of size 9, std(s_of_paths) tensor of size 9
    """
    torch.manual_seed(seed)
    theta_prior = model.prior.sample(sample_shape=(nbr_sim,))
    s_of_prior = model.model_sim(theta_prior, True)

    s_of_prior = s_of_prior.numpy()
    m_s_of_prior = stats.trim_mean(s_of_prior, cutoff, axis=0)

    lambda_trim_std = lambda s: trim_std(s, cutoff)

    s_s_of_prior = np.apply_along_axis(lambda_trim_std, 0, s_of_prior)

    if save:
        np.savetxt('data/m_s_of_prior.csv', m_s_of_prior, delimiter=",")
        np.savetxt('data/s_s_of_prior.csv', s_s_of_prior, delimiter=",")

    return m_s_of_prior, s_s_of_prior


# sets up the models
def set_up_model(seed_data=7, prior="Uniform", theta_true=torch.tensor([0.01, 0.5, 1, 0.01]).log()):
    if prior == "Uniform":
        prior_dist = BoxUniform(low=-5 * torch.ones(4), high=2 * torch.ones(4))
    else:
        Exception("Prior dist not valid!")

    # TODO add prior_dist
    # set LV model
    model = LotkaVolterra.LotkaVolterra(prior_dist)

    # gen data
    torch.manual_seed(seed_data)
    x_o = model.gen_single(theta_true)

    return x_o, model, theta_true


# TODO update w correct dims

# sets up the networks for the flow and likelihood and posterior model
def set_up_networks(seed=10, dim=4):
    torch.manual_seed(seed)
    base_dist_lik = StandardNormal(shape=[9])

    num_layers = 5

    transforms = []
    for _ in range(num_layers):  # TODO add inv sigmoide fnunc
        transforms.append(ReversePermutation(features=9))
        transforms.append(MaskedAffineAutoregressiveTransform(features=9,
                                                              hidden_features=10,
                                                              context_features=dim,
                                                              num_blocks=2))

    transform = CompositeTransform(transforms)

    flow_lik = Flow(transform, base_dist_lik)

    base_dist_post = StandardNormal(shape=[dim])

    num_layers = 4

    transforms = []

    transforms.append(PointwiseAffineTransform(shift=5 / 7.0, scale=1 / 7.0))
    transforms.append(InvSigmoid.InvSigmoid())  # this should be inv sigmoide!

    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=dim,
                                                              hidden_features=10,
                                                              context_features=9,
                                                              num_blocks=2))

    transform = CompositeTransform(transforms)

    flow_post = Flow(transform, base_dist_post)

    return flow_lik, flow_post
