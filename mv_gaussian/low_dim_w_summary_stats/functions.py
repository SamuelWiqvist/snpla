# functions for all methods

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import mv_gaussian.ConjugateMultivariateNormal as cMVN
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.permutations import ReversePermutation


# calc summary stats
def calc_summary_stats(x_input, dim=2):

    if len(x_input.shape) == 2:
        # when we only have one data set we should sum over dim 0, i.e. for the post model

        s_of_x = torch.zeros(1, 5)

        s_of_x[0, :dim] = x_input.mean(dim=0)
        cov_matrix = np.cov(x_input.detach().numpy(), rowvar=False)
        cov_upper = cov_matrix[np.triu_indices(dim)]
        s_of_x[0, dim:] = torch.from_numpy(cov_upper).flatten()

    else:
        # for mutliple data sets we sum over dim 1, i.e. for the lik model

        s_of_x = torch.zeros(x_input.shape[0], 5)

        s_of_x[:, :dim] = x_input.mean(dim=1)

        for i in range(x_input.shape[0]):
            cov_matrix = np.cov(x_input[i, :, :].detach().numpy(), rowvar=False)
            cov_upper = cov_matrix[np.triu_indices(dim)]
            s_of_x[i, dim:] = torch.from_numpy(cov_upper).flatten()

    return s_of_x


# simulate data from the model, using a fixed seed
def run_model_sim(N_samples, seed, conj_model, analytical_posterior, Sigma_real, dim=2, from_prior=True):

    if from_prior:
        torch.manual_seed(seed)
        theta = conj_model.prior.rsample(sample_shape=(N_samples,))
    else:
        torch.manual_seed(seed)
        theta = analytical_posterior.rsample(sample_shape=(N_samples,))

    torch.manual_seed(seed)

    x = torch.zeros(N_samples, conj_model.N, dim)

    for i in range(N_samples):
        model_tmp = MultivariateNormal(theta[i], Sigma_real)
        x[i, :, :] = model_tmp.rsample(sample_shape=(conj_model.N,))

    return calc_summary_stats(x), theta  # /math.sqrt(5) # div with std of prior to nomarlize data

# sets up the models
def set_up_model(seed, seed_data=10, dim=2):

    mu_prior = torch.zeros([dim])
    Sigma_prior = 5 * torch.eye(dim)
    prior = MultivariateNormal(mu_prior, Sigma_prior)

    torch.manual_seed(seed)
    mu_real = prior.rsample(sample_shape=(1,)).reshape(dim)

    torch.manual_seed(seed_data)
    a = torch.randn(dim, dim) * dim / 2
    Sigma_real = torch.mm(a, a.t())  # make symmetric positive-definite
    Sigma_real_tril = torch.cholesky(Sigma_real)

    # Set model
    model = MultivariateNormal(loc=mu_real, scale_tril=Sigma_real_tril)

    # set conj model
    conj_model = cMVN.ConjugateMultivariateNormalSummayStats(model, prior, 500, dim)

    # generate data
    x_o = conj_model.sample_fixed_seed(seed)

    # Analytical posterior

    analytical_posterior = conj_model.calc_analytical_posterior(x_o)

    return x_o, conj_model, analytical_posterior


# sets up the networks for the flow and likelihood and posterior model
def set_up_networks(seed=10, dim=2):

    torch.manual_seed(seed)
    base_dist_lik = StandardNormal(shape=[5])

    num_layers = 5

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=5))
        transforms.append(MaskedAffineAutoregressiveTransform(features=5,
                                                              hidden_features=10,
                                                              context_features=dim,
                                                              num_blocks=2))

    transform = CompositeTransform(transforms)

    flow_lik = Flow(transform, base_dist_lik)

    base_dist_post = StandardNormal(shape=[dim])

    num_layers = 4

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=dim,
                                                              hidden_features=10,
                                                              context_features=5,
                                                              num_blocks=2))

    transform = CompositeTransform(transforms)

    flow_post = Flow(transform, base_dist_post)

    return flow_lik, flow_post

def sample_hp(method, case):

    torch.manual_seed(case)

    if method == "snpe_c" or method == "snl" or method == "snre_b":
        return 10 ** -4 + (10 ** -2 - 10 ** -4) * torch.rand(1)
    else:
        lr_like = 10 ** -4 + (10 ** -2 - 10 ** -4) * torch.rand(1)
        lr_post = 10 ** -4 + (10 ** -2 - 10 ** -4) * torch.rand(1)
        gamma_post = 0.8 + (0.999 - 0.8) * torch.rand(1)
        lam = 0.65 + (0.95 - 0.65) * torch.rand(1)
        return [lr_like[0].item(), lr_post[0].item(), gamma_post[0].item(), lam[0].item()]

