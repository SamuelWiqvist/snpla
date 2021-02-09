# Imports
import sys
import torch
import os
from sbi.utils import BoxUniform
import copy
import time
import numpy as np

# Initial set up
lunarc = int(sys.argv[1])
dim = int(sys.argv[2])
seed = int(sys.argv[3])
seed_data = int(sys.argv[4])

print("Input args:")
print("Dim: " + str(dim))
print("seed: " + str(seed))
print("seed_data: " + str(seed_data))

# Set wd
print(os.getcwd())

# Set wd
print(os.getcwd())

# set the wd to the base folder for the project
if lunarc == 1:
    os.chdir('/home/samwiq/spa/seq-posterior-approx-w-nf-dev')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev')

sys.path.append('./')

print(os.getcwd())

id_job = str(dim) + '_' + str(seed) + '_' + str(seed_data)


import two_moons.functions as func
import algorithms.spa as spa

prior = BoxUniform(low=-2 * torch.ones(2), high=2 * torch.ones(2))
x_o, model = func.set_up_model(prior)
dim = 2

flow_lik, flow_post = func.set_up_networks(seed)

optimizer_lik = torch.optim.Adam(flow_lik.parameters())
optimizer_post = torch.optim.Adam(flow_post.parameters(), lr=10 ** -3)
decay_rate_post = 0.95

# test prior pred sampling and sampling for given that

nbr_rounds = 5
prob_prior_decay_rate = 0.8
prob_prior = spa.calc_prob_prior(nbr_rounds, prob_prior_decay_rate)

print(prob_prior_decay_rate)

nbr_lik = [2000, 2000, 2000, 2000, 2000]
nbr_epochs_lik = [50, 50, 50, 50, 50]
batch_size = 2000
batch_size_post = 2000
nbr_post = [40000, 40000, 40000, 40000, 40000]
nbr_epochs_post = [50, 50, 50, 50, 50]

x_o_batch_post = torch.zeros(batch_size_post, 2)

for i in range(batch_size_post):
    x_o_batch_post[i, :] = x_o

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start = time.time()

# TODO check prior and model sim
models_lik, models_post = spa.inference_spa(flow_lik,
                                            flow_post,
                                            model.prior,
                                            model.model_sim,
                                            optimizer_lik,
                                            optimizer_post,
                                            decay_rate_post,
                                            x_o.reshape(1, dim),
                                            x_o_batch_post,
                                            dim,
                                            prob_prior,
                                            nbr_lik,
                                            nbr_epochs_lik,
                                            nbr_post,
                                            nbr_epochs_post,
                                            batch_size,
                                            batch_size_post)

end = time.time()
run_time = end - start
print(run_time)

torch.manual_seed(seed)

start = time.time()

for i in range(nbr_rounds):
    theta_trained = models_post[i].sample(1000, context=x_o.reshape(1, dim))
    theta_trained = theta_trained.reshape((1000, 2))

    np.savetxt('two_moons/data/spa_flow_posterior_' + str(i + 1) + "_" + str(dim) + '_' + str(seed) + '.csv',
               theta_trained.detach().numpy(), delimiter=",")
#id_job = str(dim) + '_' + str(seed) + '_' + str(seed_data)

end = time.time()
run_time_inference = (end - start) / nbr_rounds

torch.manual_seed(seed)

theta_test = model.gen_posterior_samples(x_o, 1000)

for i in range(nbr_rounds):
    x_trained = models_lik[i].sample(1, context=theta_test)
    x_trained = x_trained.reshape((1000, 2))

    np.savetxt('two_moons/data/spa_flow_post_pred_' + str(i + 1) + "_" + id_job + '.csv',
               x_trained.detach().numpy(), delimiter=",")

with open('two_moons/results/spa_flow_' + id_job + '.txt', 'w') as f:
    f.write('%.4f\n' % run_time)
    f.write('%.4f\n' % run_time_inference)
