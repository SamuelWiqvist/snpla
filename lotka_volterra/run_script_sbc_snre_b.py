# Imports
import sys
import torch
import os
import numpy as np
import time
from sbi.inference import SNRE_B, prepare_for_sbi

# Initial set up
lunarc = int(sys.argv[1])
seed = int(sys.argv[2])

print("Input args:")
print("seed: " + str(seed))

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


inference = SNRE_B(simulator, prior)

learning_rate = 0.0005  # default value

start = time.time()

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

prior_samples = prior.sample(sample_shape=(1,))
data_sets = simulator(prior_samples)

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
K = 4
M = L
Lprime = 50

run_mcmc = True
post_samples = posteriors[-1].sample((Lprime,), x=x_o)

print(post_samples)
ess_current = func.ess_mcmc(post_samples)
print(ess_current)

if ess_current < M:
    # continu mcmc
    L_all = int(Lprime * M / ess_current)
    print(L_all)
    post_samples = posteriors[-1].sample((L_all,), x=x_o)
    # post_samples = torch.cat((post_samples, post_samples_new))
    # else:
    # run_mcmc = False

# thinning chain
ess_current = func.ess_mcmc(post_samples)
print(ess_current)

N_total = post_samples.shape[0]
post_samples = post_samples[range(0, N_total, int(N_total / M)), :]  # thin samples

indications = torch.zeros(K)

for k in range(4):
    indications[k] = (post_samples[:, k] < prior_samples[0, k]).sum()

np.savetxt('sbc/ranks_snre_b_' + id_job + '.csv', indications.numpy(), delimiter=",")
