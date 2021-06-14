# Imports
import sys
import torch
import os
import time
from sbi.inference import SNLE_A, prepare_for_sbi
import numpy as np
import math

# Initial set up
lunarc = int(sys.argv[1])
seed = int(sys.argv[2])
hp_tuning = int(sys.argv[3])  # if hp_tuning = 0, no hyper-param tuning, else hp_tuning for that sample of the hp
seed_data = 7

print("Input args:")
print("seed: " + str(seed))
print("seed_data: " + str(seed_data))

# Set wd
print(os.getcwd())

# set the wd to the base folder for the project
if lunarc == 1:
    os.chdir('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev/lotka_volterra')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/lotka_volterra')

sys.path.append('./')

print(os.getcwd())

id_job = str(seed) + '_' + str(seed_data)

if hp_tuning > 0:
    id_job = id_job + "_" + str(hp_tuning)


# Load all utility functions for all methods
import LotkaVolterra
import functions as func  # Set model and generate data

print(hp_tuning)
print(func.sample_hp("snl", hp_tuning))
print(torch.rand(1))
print(func.sample_hp("snl", hp_tuning))
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
def build_custom_like_net(batch_theta, batch_x):
    flow_lik, flow_post = func.set_up_networks()

    return flow_lik


inference = SNLE_A(simulator, prior, density_estimator=build_custom_like_net)

hyper_params = [0.0005, 0.98]  # default value

if hp_tuning >= 2:
    hyper_params = func.sample_hp("snl", hp_tuning)


start = time.time()

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_rounds = 5
x_o = s_x_o

posteriors = []
proposal = None

print(hyper_params)

for i in range(num_rounds):
    learning_rate = hyper_params[0]*math.exp(-hyper_params[1] * i)
    posterior = inference(num_simulations=1000, proposal=proposal, max_num_epochs=50, learning_rate=learning_rate)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

end = time.time()
run_time = end - start

print("")
print("Runtime:" + str(round(run_time, 2)))

# gen posterior samples

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

log_probs = []
N_post_pred_test = 1000
theta_post_pred = torch.zeros((N_post_pred_test, 4))

for i in range(num_rounds):

    posterior_sample = posteriors[i].sample((1000,), x=x_o)  # TODO add gaussian approx??
    post_gauss_approx = func.fit_gaussian_dist(posterior_sample)
    #print(-post_gauss_approx.log_prob(theta_true))
    log_probs.append(-post_gauss_approx.log_prob(theta_true))

    #log_probs.append(-posteriors[i].log_prob(theta_true, x=x_o))

    if i == num_rounds - 1:
        print(posterior_sample.shape)
        theta_post_pred = posterior_sample

    np.savetxt('data/snl_posterior_' + str(i + 1) + "_" + id_job + '.csv',
               posterior_sample.detach().numpy(), delimiter=",")

    if hp_tuning == 0:

        np.savetxt('data/snl_posterior_' + str(i + 1) + "_" + id_job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")

    else:

        np.savetxt('hp_tuning/snl_posterior_' + str(i + 1) + "_" + id_job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")


end = time.time()
run_time_inference = (end - start) / num_rounds

# Write results

#with open('results/snl_' + id_job + '.txt', 'w') as f:
#    f.write('%.4f\n' % run_time)
#    f.write('%.4f\n' % run_time_inference)
#    for i in range(num_rounds):
#        f.write('%.4f\n' % log_probs[i])

if hp_tuning == 0:

    with open('results/snl_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        for i in range(num_rounds):
            f.write('%.4f\n' % log_probs[i])

else:

    with open('hp_tuning/snl_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % hp_tuning)
        f.write('%.6f\n' % hyper_params[0])
        f.write('%.6f\n' % hyper_params[1])
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        for i in range(num_rounds):
            f.write('%.4f\n' % log_probs[i])

# gen samples from likelihood model
if hp_tuning == 0:


    # prior pred samples
    torch.manual_seed(seed + 1)
    N_prior_pred_test = 1000
    theta_test = model.prior.rsample(sample_shape=(N_prior_pred_test,))

    # obs data samples
    torch.manual_seed(seed + 2)
    N_test_obs_data = 1000
    theta_test_obs_data = torch.zeros(N_test_obs_data, 4)

    for i in range(N_test_obs_data):
        theta_test_obs_data[i, :] = theta_true

    theta_test = theta_test.reshape(N_prior_pred_test, 4)
    theta_test_obs_data = theta_test_obs_data.reshape(N_test_obs_data, 4)
    theta_post_pred = theta_post_pred.reshape(N_post_pred_test, 4)

    # gen samples from trained model
    torch.manual_seed(seed)
    x_prior = inference._posterior.net.sample(1, context=theta_test)
    x_theta_true = inference._posterior.net.sample(1, context=theta_test_obs_data)
    x_post = inference._posterior.net.sample(1, context=theta_post_pred)

    x_prior = x_prior.reshape((N_prior_pred_test, 9))
    x_theta_true = x_theta_true.reshape((N_test_obs_data, 9))
    x_post = x_post.reshape((N_post_pred_test, 9))

    # Write results
    np.savetxt('data/data_recon_prior_snl_' + id_job + '.csv', x_prior.detach().numpy(), delimiter=",")

    np.savetxt('data/data_recon_snl_' + id_job + '.csv', x_theta_true.detach().numpy(), delimiter=",")

    np.savetxt('data/data_recon_post_snl_' + id_job + '.csv', x_post.detach().numpy(), delimiter=",")
