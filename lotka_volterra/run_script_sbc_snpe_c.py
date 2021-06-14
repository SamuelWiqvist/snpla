# Imports
import sys
import torch
import os
import time
from sbi.inference import SNPE_C, prepare_for_sbi
import numpy as np

# Initial set up
lunarc = int(sys.argv[1])
seed = int(sys.argv[2])

print("Input args:")
print("seed: " + str(seed))

# Set wd
print(os.getcwd())

# set the wd to the base folder for the project
if lunarc == 1:
    os.chdir('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev/lotka_volterra')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/lotka_volterra')

sys.path.append('./')

print(os.getcwd())

id_job = str(seed)

import LotkaVolterra
import functions as func  # Set model and generate data

# Set model and generate data

x_o, model, theta_true = func.set_up_model()
m_s_of_prior, s_s_of_prior = func.load_summary_stats_mean_and_std()

# set up simulator
def simulator(theta):
    s_of_theta = model.model_sim(theta)

    return func.normalize_summary_stats(s_of_theta, m_s_of_prior, s_s_of_prior)


s_x_o = LotkaVolterra.calc_summary_stats(x_o.reshape(1, x_o.shape[0], x_o.shape[1]))
s_x_o = func.normalize_summary_stats(s_x_o, m_s_of_prior, s_s_of_prior)

# check simulator and prior
simulator, prior = prepare_for_sbi(simulator, model.prior)


# function that builds the network
def build_custom_post_net(batch_theta, batch_x):
    flow_lik, flow_post = func.set_up_networks()

    return flow_post

inference = SNPE_C(simulator, prior, density_estimator=build_custom_post_net)

learning_rate = 0.0005  # default value

start = time.time()

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

prior_samples = prior.sample(sample_shape=(1,))
data_sets = simulator(prior_samples)

print(prior_samples)
print(data_sets)
print(data_sets.shape)
print(prior_samples.shape)

num_rounds = 5
x_o = data_sets

posteriors = []
proposal = None

print(learning_rate)

for i in range(num_rounds):
    posterior = inference(num_simulations=1000, proposal=proposal, max_num_epochs=50, learning_rate=learning_rate)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

end = time.time()
run_time = end - start

print("")
print("Runtime:" + str(round(run_time, 2)))

L = 4
M = L
K = 4

indications = torch.zeros(4)

post_samples = posteriors[-1].sample((M,), x=x_o)

for k in range(K):
    indications[k] = (post_samples[:, k] < prior_samples[0, k]).sum()

np.savetxt('sbc/ranks_snpec_' + id_job + '.csv', indications.numpy(), delimiter=",")
