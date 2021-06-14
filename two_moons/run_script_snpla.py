# Imports
import sys
import torch
import os
from torch.distributions.uniform import Uniform

import copy
import time
import numpy as np
import random

# TODO: the method is mode seeking! So once we have found the modes the training of the post model can go of the rails,
#  sicne it cannot realy imporce from where it is! This also explines why it is doing not so great for TM!

# Reg:
# 1) large batch for post model - this should indeed be the case
# 2) L2 reg - seems ok
# 3) large network for post - not sure about this...
# 4) learn rate decay
# 5) mode seeking - why we have particular problems for TM!
# 7) jumpt where the loss is increasing!
# 8) more, smaller steps in the algo

# 1) flux from loss -> large batch size + lr decay + l2 reg etc
# 2) flux from mode-seeking -> not sure how to fix that

# Initial set up
lunarc = int(sys.argv[1])
dim = int(sys.argv[2])
seed = int(sys.argv[3])
seed_data = int(sys.argv[4])
hp_tuning = int(sys.argv[5])  # if hp_tuning = 0, no hyper-param tuning, else hp_tuning for that sample of the hp
lambda_val = float(sys.argv[6])  # if hp_tuning = 0, no hyper-param tuning, else hp_tuning for that sample of the hp

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
    os.chdir('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev')

sys.path.append('./')

print(os.getcwd())

id_job = str(dim) + '_' + str(seed) + '_' + str(seed_data)

if hp_tuning > 0:
    id_job = id_job + "_" + str(hp_tuning)

if lambda_val > 0:
    id_job = id_job + "_" + str(lambda_val)

import two_moons.functions as func
import algorithms.snpla as snpla

prior = Uniform(low=-2 * torch.ones(2), high=2 * torch.ones(2))
x_o, model = func.set_up_model(prior)
dim = 2

flow_lik, flow_post = func.set_up_networks(seed)

hyper_params = [0.001, 0.001, 0.9, 0.7]  # lr_like, lr_post, gamma_post, gamma_lik, gamma

if lambda_val > 0:
    hyper_params[-1] = lambda_val

# hyper_params = [0.001, 0.001, 0.95, 0.0, 0.8] good res!

if hp_tuning >= 2:
    hyper_params = func.sample_hp("snpla", hp_tuning)

optimizer_lik = torch.optim.Adam(flow_lik.parameters(), lr=hyper_params[0], weight_decay=0.0)
optimizer_post = torch.optim.Adam(flow_post.parameters(), lr=hyper_params[1], weight_decay=0.0)
decay_rate_post = hyper_params[2]

# test prior pred sampling and sampling for given that

nbr_rounds = 10
prob_prior_decay_rate = hyper_params[3]
prob_prior = snpla.calc_prob_prior(nbr_rounds, prob_prior_decay_rate)

print(prob_prior_decay_rate)
print(prob_prior)

# nbr_lik = [2000, 2000, 2000, 2000, 2000]
# nbr_epochs_lik = [50, 50, 50, 50, 50]
# batch_size = 100
# batch_size_post = 100
# nbr_post = [20000, 20000, 20000, 20000, 20000]
# nbr_epochs_post = [25, 25, 25, 25, 25]

print(hyper_params)

nbr_lik = [1000 for _ in range(nbr_rounds)]  # [1000, 1000, 1000, 1000, 1000]  # , 2000, 2000]
nbr_epochs_lik = [25 for _ in range(nbr_rounds)]  # [100, 100, 100, 100, 100]
batch_size = 2000  # this is really important due to the two modes of the posterior, ow prob not so important
batch_size_post = 20000
nbr_post = [60000 for _ in range(nbr_rounds)]  # [10000, 10000, 10000, 10000, 10000]  # , 10000, 10000]
nbr_epochs_post = [75 for _ in range(nbr_rounds)]  # [50, 50, 50, 50, 50, 50]

x_o_batch_post = torch.zeros(batch_size_post, 2)

for i in range(batch_size_post):
    x_o_batch_post[i, :] = x_o

random.seed(seed, version=2)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start = time.time()

# TODO check prior and model sim
models_lik, models_post = snpla.inference_snpla(flow_lik,
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

random.seed(seed, version=2)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start = time.time()

for i in range(nbr_rounds):
    theta_trained = models_post[i].sample(1000, context=x_o.reshape(1, dim))
    theta_trained = theta_trained.reshape((1000, 2))

    if hp_tuning == 0 and lambda_val > 0:

        np.savetxt('two_moons/lambda_val/post_samples_snpla_' + str(i + 1) + "_" + id_job + '.csv',
                   theta_trained.detach().numpy(), delimiter=",")

    elif hp_tuning == 0:

        np.savetxt('two_moons/data/snpla_posterior_' + str(i + 1) + "_" + id_job + '.csv',
                   theta_trained.detach().numpy(), delimiter=",")

    else:

        np.savetxt('two_moons/hp_tuning/post_samples_snpla_' + str(i + 1) + "_" + id_job + '.csv',
                   theta_trained.detach().numpy(), delimiter=",")

end = time.time()
run_time_inference = (end - start) / nbr_rounds

if hp_tuning == 0 and lambda_val > 0:

    with open('two_moons/lambda_val/snpla_' + id_job + '.txt', 'w') as f:
        for h in hyper_params:
            f.write('%.6f\n' % h)
        for p in prob_prior:
            f.write('%.6f\n' % p)
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)

elif hp_tuning == 0:

    with open('two_moons/results/snpla_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)

else:

    with open('two_moons/hp_tuning/snpla_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % hp_tuning)
        for h in hyper_params:
            f.write('%.6f\n' % h)
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)

if hp_tuning == 0:
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    random.seed(seed, version=2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    theta_test = model.gen_posterior_samples(x_o, 1000)

    for i in range(nbr_rounds):
        x_trained = models_lik[i].sample(1, context=theta_test)
        x_trained = x_trained.reshape((1000, 2))

        np.savetxt('two_moons/data/snpla_post_pred_' + str(i + 1) + "_" + id_job + '.csv',
                   x_trained.detach().numpy(), delimiter=",")

    with open('two_moons/results/snpla_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
