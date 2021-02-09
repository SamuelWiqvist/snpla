# functions for all methods

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import mv_gaussian.ConjugateMultivariateNormal as cMVN
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.permutations import ReversePermutation


class SummaryNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_inner_f3 = 20
        self.hidden_inner_f2 = 20
        self.hidden_inner_f1 = 20
        self.dim_out_1 = 5
        self.x_dim = 2
        self.n_obs = 5

        self.summary_net_f1 = nn.Sequential(nn.Linear(self.dim_out_1, self.hidden_inner_f1),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_inner_f1, self.dim_out_1))

        self.summary_net_f2 = nn.Sequential(nn.Linear(self.x_dim, self.hidden_inner_f2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_inner_f2, self.dim_out_1))

        self.summary_net_f3 = nn.Sequential(nn.Linear(self.x_dim, self.hidden_inner_f3),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_inner_f3, self.dim_out_1))

    def forward(self, x):  # forward function for summary net

        nbr_samples = x.shape[0]

        x = x.reshape((nbr_samples, self.n_obs, self.x_dim))

        #w_tilde_i = self.summary_net_f3(x)
        #w_i = torch.exp(w_tilde_i) / torch.sum(torch.exp(w_tilde_i), 0)

        f2_out = self.summary_net_f2(x)

        #f1_input = torch.sum(w_i * f2_out, 1)
        f1_input = torch.sum(f2_out, 1)
        s_of_x = self.summary_net_f1(f1_input)

        return s_of_x.reshape(nbr_samples, self.dim_out_1)


def flatten(x):
    return torch.reshape(x, (x.shape[0], -1))


# simulate data from the model
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

    # return calc_summary_stats(x), theta #/math.sqrt(5) # div with std of prior to nomarlize data
    return flatten(x), theta


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
    conj_model = cMVN.ConjugateMultivariateNormal(model, prior, 5, 2 * 5)

    # generate data
    x_o = conj_model.sample_fixed_seed(seed)

    # Analytical posterior

    analytical_posterior = conj_model.calc_analytical_posterior(x_o)

    return x_o, conj_model, analytical_posterior


# sets up the networks for the flow and likelihood and posterior model
# in case is summary net not included in the post flow model
def set_up_networks(seed=10, dim=2):
    torch.manual_seed(seed)

    base_dist = StandardNormal(shape=[10])

    num_layers = 4

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=10))
        transforms.append(MaskedAffineAutoregressiveTransform(features=10,
                                                              hidden_features=40,
                                                              context_features=dim,
                                                              num_blocks=1))

    transform = CompositeTransform(transforms)

    flow_lik = Flow(transform, base_dist)

    base_dist_post = StandardNormal(shape=[dim])

    num_layers = 4

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=dim,
                                                              hidden_features=40,
                                                              context_features=5,  # it sort of makes more sense that
                                                              # context_features is the nbr of features returned by the
                                                              # summary net
                                                              num_blocks=1))

    transform = CompositeTransform(transforms)

    # here we add the SummaryNet as an embedded network to the flow model for the posterior
    flow_post = Flow(transform, base_dist_post, embedding_net=SummaryNet())

    return flow_lik, flow_post

