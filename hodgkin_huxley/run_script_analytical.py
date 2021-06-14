# Imports
import sys
import torch
import os
import numpy as np

# Initial set up
#lunarc = int(sys.argv[1])
#dim = int(sys.argv[2])
#seed = int(sys.argv[3])
#seed_data = int(sys.argv[4])

#print("Input args:")
##print("Dim: " + str(dim))
#print("seed: " + str(seed))
#print("seed_data: " + str(seed_data))

lunarc = int(sys.argv[1])
seed = int(sys.argv[2])
seed_data = 7


# Set wd
print(os.getcwd())

# set the wd to the base folder for the project
if lunarc == 1:
    os.chdir('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev/lotka_volterra')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/lotka_volterra')

sys.path.append('./')

print(os.getcwd())

# Load all utility functions for all methods
import LotkaVolterra
import functions as func# Set model and generate data

# Set model and generate data

x_o, model, theta_true = func.set_up_model()
m_s_of_prior, s_s_of_prior = func.load_summary_stats_mean_and_std()


# set up simulator
def simulator(theta):
    s_of_theta = model.model_sim(theta)

    return func.normalize_summary_stats(s_of_theta, m_s_of_prior, s_s_of_prior)


s_x_o = LotkaVolterra.calc_summary_stats(x_o.reshape(1, x_o.shape[0], x_o.shape[1]))
s_x_o = func.normalize_summary_stats(s_x_o, m_s_of_prior, s_s_of_prior)

## Generate test data
# prior pred samples
torch.manual_seed(seed+1)

N_prior_pred_test = 1000

theta_test = model.prior.rsample(sample_shape=(N_prior_pred_test,))
s_of_theta_test = model.model_sim(theta_test, True)
x_prior = func.normalize_summary_stats(s_of_theta_test, m_s_of_prior, s_s_of_prior)

torch.manual_seed(seed+2)
# obs data samples
N_test_obs_data = 1000

theta_test_obs_data = torch.zeros(N_test_obs_data, 4)

for i in range(N_test_obs_data):

    theta_test_obs_data[i, :] = theta_true

s_of_theta_test_obs_data = model.model_sim(theta_test_obs_data, True)
x_test_obs_data = func.normalize_summary_stats(s_of_theta_test_obs_data, m_s_of_prior, s_s_of_prior)

id_job = str(seed) + '_' + str(seed_data)

# Write results
np.savetxt('data/data_recon_prior_model_' + id_job + '.csv', x_prior.detach().numpy(), delimiter=",")

np.savetxt('data/data_recon_model_' + id_job+ '.csv', x_test_obs_data.detach().numpy(), delimiter=",")

