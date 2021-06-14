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
hp_tuning = int(sys.argv[3])  # if hp_tuning = 0, no hyper-param tuning, else hp_tuning for that sample of the hp

seed_data = 7

print("Input args:")
print("seed: " + str(seed))
print("seed_data: " + str(seed_data))

if lunarc == 1:
    os.chdir('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev/lotka_volterra')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/lotka_volterra')

sys.path.append('./')

print(os.getcwd())

id_job = str(seed) + '_' + str(seed_data)

if hp_tuning > 0:
    id_job = id_job + "_" + str(hp_tuning)

import LotkaVolterra
import functions as func  # Set model and generate data

print(hp_tuning)
print(func.sample_hp("snre_b", hp_tuning))
print(torch.rand(1))
print(func.sample_hp("snre_b", hp_tuning)[0].item())
print(torch.rand(1))

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

if hp_tuning >= 2:
    learning_rate = func.sample_hp("snre_b", hp_tuning)[0].item()

start = time.time()

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_rounds = 5

x_o = s_x_o

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

start = time.time()
log_probs = []

for i in range(num_rounds):
    print(i)
    posterior_sample = posteriors[i].sample((1000,), x=x_o)
    # log_probs.append(-posteriors[i].log_prob(theta_true, x=x_o))
    post_gauss_approx = func.fit_gaussian_dist(posterior_sample)  # to get correct prob
    log_probs.append(-post_gauss_approx.log_prob(theta_true))

    if hp_tuning == 0:

        np.savetxt('data/post_samples_snre_b_' + str(i + 1) + "_" + id_job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")

    else:

        np.savetxt('hp_tuning/post_samples_snre_b_' + str(i + 1) + "_" + id_job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")

end = time.time()
run_time_inference = (end - start) / num_rounds

# Write results

if hp_tuning == 0:

    with open('results/snre_b_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        for i in range(num_rounds):
            f.write('%.4f\n' % log_probs[i])

else:

    with open('hp_tuning/snre_b_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % hp_tuning)
        f.write('%.6f\n' % learning_rate)
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        for i in range(num_rounds):
            f.write('%.4f\n' % log_probs[i])

start = time.time()

