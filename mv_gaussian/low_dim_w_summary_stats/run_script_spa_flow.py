# Imports
import sys
import torch
import os
import time
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

# Initial set up
lunarc = int(sys.argv[1])
dim = int(sys.argv[2])
seed = int(sys.argv[3])
seed_data = int(sys.argv[4])

print("Input args:")
print("Dim: " + str(dim))
print("seed: " + str(seed))
print("seed_data: " + str(seed_data))


id_job = str(dim) + '_' + str(seed) + '_' + str(seed_data)

# Set wd
print(os.getcwd())

# set the wd to the base folder for the project
if lunarc == 1:
    os.chdir('/home/samwiq/spa/seq-posterior-approx-w-nf-dev')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev')

sys.path.append('./')

print(os.getcwd())

# Load all utility functions for all methods
import mv_gaussian.low_dim_w_summary_stats.functions as func
import algorithms.spa as spa

# Set model and generate data

x_o, conj_model, analytical_posterior = func.set_up_model(seed)

# set up posterior network
flow_lik, flow_post = func.set_up_networks()

## Generate test data

N_prior_pred_test = 1000
x_test, theta_test = func.run_model_sim(N_prior_pred_test, seed + 2, conj_model, analytical_posterior,
                                                    conj_model.model.covariance_matrix, dim, True)

# Generate test data for obs data set
print(conj_model.model_sim(theta_test).shape)


N_test_obs_data = 1000

x_test_obs_data = torch.zeros(N_test_obs_data, 5)
theta_test_obs_data = torch.zeros(N_test_obs_data, dim)

for i in range(N_test_obs_data):

    x_test_obs_data[i, :] = func.calc_summary_stats(x_o)
    theta_test_obs_data[i, :] = conj_model.model.loc

# Set up networks for the likelihood model

# Base dist for posterior model
flow_lik, flow_post = func.set_up_networks()

optimizer_lik = torch.optim.Adam(flow_lik.parameters())
optimizer_post = torch.optim.Adam(flow_post.parameters(), lr=0.002)
decay_rate_post = 0.95  # no adaptation of Adam's base rate

nbr_rounds = 10
prob_prior_decay_rate = 0.7
prob_prior = spa.calc_prob_prior(nbr_rounds, prob_prior_decay_rate)

print(prob_prior)

#nbr_lik = [2000, 2000, 2000, 2000]
#nbr_epochs_lik = [25, 25, 25, 25]
#batch_size = 50
#batch_size_post = 50
#nbr_post = [10000, 10000, 10000, 10000]
#nbr_epochs_post = [25, 25, 25, 25]


nbr_lik = [2500 for _ in range(nbr_rounds)]  # [1000, 1000, 1000, 1000, 1000]  # , 2000, 2000]
nbr_epochs_lik = [75 for _ in range(nbr_rounds)]  # [100, 100, 100, 100, 100]
batch_size = 50
batch_size_post = 50
nbr_post = [10000 for _ in range(nbr_rounds)]  # [10000, 10000, 10000, 10000, 10000]  # , 10000, 10000]
nbr_epochs_post = [75 for _ in range(nbr_rounds)]  # [50, 50, 50, 50, 50, 50]


x_o_batch_post = torch.zeros(batch_size_post, 5)

for i in range(batch_size_post):
    x_o_batch_post[i, :] = func.calc_summary_stats(x_o)

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start = time.time()

# TODO check prior and simulator
models_lik, models_post = spa.inference_spa(flow_lik,
                                            flow_post,
                                            conj_model.prior,
                                            conj_model.model_sim,
                                            optimizer_lik,
                                            optimizer_post,
                                            decay_rate_post,
                                            func.calc_summary_stats(x_o),
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

print("")
print("Runtime:" + str(round(run_time, 2)))

kl_divs_trained = []
start = time.time()
torch.manual_seed(seed)

for i in range(nbr_rounds):
    print(i)
    posterior_sample = models_post[i].sample(1000, context=func.calc_summary_stats(x_o))
    posterior_sample = posterior_sample.reshape((1000, 2))

    kl_divs_trained.append(conj_model.kl_div(analytical_posterior, posterior_sample))

    np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/post_samples_spa_flow_' + str(i + 1) + "_" + id_job + '.csv',
               posterior_sample.detach().numpy(), delimiter=",")

end = time.time()
run_time_inference = (end - start) / nbr_rounds

# Inference

# Sample data from post pred
N_post_pred_test = 1000
x_post_pred, theta_post_pred = func.run_model_sim(N_post_pred_test, seed + 3, conj_model, analytical_posterior,
                                                  conj_model.model.covariance_matrix, dim, False)

torch.manual_seed(seed)
x_prior = flow_lik.sample(1, context=theta_test)
x_theta_true = flow_lik.sample(1, context=theta_test_obs_data)
x_post = flow_lik.sample(1, context=theta_post_pred)

x_prior = x_prior.reshape(x_test.shape)
x_theta_true = x_theta_true.reshape(x_test_obs_data.shape)
x_post = x_post.reshape(x_post_pred.shape)


# Write results

with open('mv_gaussian/low_dim_w_summary_stats/results/spa_flow_' + id_job + '.txt', 'w') as f:

    f.write('%.4f\n' % run_time)
    f.write('%.4f\n' % run_time_inference)
    for i in range(nbr_rounds):
        f.write('%.4f\n' % kl_divs_trained[i])

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/data_recon_spa_flow_' + id_job +
           '.csv', x_theta_true.detach().numpy(), delimiter=",")

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/data_recon_prior_spa_flow_' + id_job + '.csv',
           x_prior.detach().numpy(), delimiter=",")

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/data_recon_post_spa_flow_' + id_job + '.csv',
           x_post.detach().numpy(), delimiter=",")
