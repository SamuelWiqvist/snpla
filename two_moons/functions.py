# functions for all methods
import torch
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.standard import PointwiseAffineTransform
import TwoMoons

from nflows.transforms.base import (
    CompositeTransform,
)

# load from util (from https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from util import InvSigmoid


# sets up the models
def set_up_model(prior, mean_radius=1.0, sd_radius=0.1, baseoffset=1.0, fixed_data=True):
    model = TwoMoons.TwoMoons(prior, mean_radius, sd_radius, baseoffset, fixed_data)

    if fixed_data:
        x_o = torch.zeros(2)

    return x_o, model


# sets up the networks for the flow and likelihood and posterior model
def set_up_networks(seed=10, dim=2):
    torch.manual_seed(seed)
    base_dist_lik = StandardNormal(shape=[2])

    num_layers = 5

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                              hidden_features=50,
                                                              context_features=dim,
                                                              num_blocks=1))

    transform = CompositeTransform(transforms)

    flow_lik = Flow(transform, base_dist_lik)

    base_dist_post = StandardNormal(
        shape=[dim])  # BoxUniform(low=-2*torch.ones(2), high=2*torch.ones(2)) #StandardNormal(shape=[dim])

    # base_dist_post = BoxUniform(low=-2*torch.ones(2), high=2*torch.ones(2))

    num_layers = 5

    transforms = []

    transforms.append(PointwiseAffineTransform(shift=0.5, scale=1 / 4.0))
    transforms.append(InvSigmoid.InvSigmoid())  # this should be inv sigmoide!

    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=dim,
                                                              hidden_features=50,
                                                              context_features=2,
                                                              num_blocks=1))

    transform = CompositeTransform(transforms)

    flow_post = Flow(transform, base_dist_post)

    return flow_lik, flow_post
