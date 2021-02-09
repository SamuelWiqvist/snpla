# Imports
import sys
import torch
import os
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

# set the wd to the base folder for the project
if lunarc == 1:
    os.chdir('/home/samwiq/spa/seq-posterior-approx-w-nf-dev')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev')

sys.path.append('./')

print(os.getcwd())

# Load all utility functions for all methods
import mv_gaussian.low_dim_w_summary_stats.functions as func

# Set model and generate data

x_o, conj_model, analytical_posterior = func.set_up_model(seed)

# sample data from posterior
torch.manual_seed(seed)
analytical_posterior_samples = conj_model.sample_analytical_posterior(analytical_posterior, 1000)

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/post_samples_analytical_' + str(dim) + '_' + str(seed) + '_' +
           str(seed_data) + '.csv', analytical_posterior_samples.detach().numpy(), delimiter=",")

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/x_o_summary_stats_true_' + str(dim) + '_' + str(seed) + '_' +
           str(seed_data) + '.csv', func.calc_summary_stats(x_o).detach().numpy(), delimiter=",")

# Sample data sets for ground-truth model
N_test_obs_data = 1000

x_test_obs_data = torch.zeros(N_test_obs_data, 5)
theta_test_obs_data = torch.zeros(N_test_obs_data, dim)

for i in range(N_test_obs_data):
    x_test_obs_data[i, :] = func.calc_summary_stats(conj_model.sample())

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/x_o_summary_stats_samples_true_' + str(dim) + '_' + str(seed) +
           '_' + str(seed_data) + '.csv', x_test_obs_data.detach().numpy(), delimiter=",")

# Sample data from prior pred
N_prior_pred_test = 1000
x_prior_pred, theta_prior_pred = func.run_model_sim(N_prior_pred_test, seed + 2, conj_model, analytical_posterior,
                                                    conj_model.model.covariance_matrix, dim, True)

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/x_summary_stats_samples_prior_pred_' + str(dim) + '_' + str(seed) +
           '_' + str(seed_data) + '.csv', x_prior_pred.detach().numpy(), delimiter=",")

# Sample data from post pred
N_post_pred_test = 1000
x_post_pred, theta_post_pred = func.run_model_sim(N_post_pred_test, seed + 3, conj_model, analytical_posterior,
                                                  conj_model.model.covariance_matrix, dim, False)

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/x_summary_stats_samples_post_pred_' + str(dim) + '_' + str(seed) +
           '_' + str(seed_data) + '.csv', x_post_pred.detach().numpy(), delimiter=",")
