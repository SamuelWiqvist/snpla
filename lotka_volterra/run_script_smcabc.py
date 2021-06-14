# Imports
import sys
import torch
import os
import numpy as np
import time
from sbi.inference import SMCABC, prepare_for_sbi

# Initial set up
lunarc = int(sys.argv[1])
seed = int(sys.argv[2])
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

# Load all utility functions for all methods
import LotkaVolterra
import functions as func

# Set model and generate data

x_o, model, theta_true = func.set_up_model()
m_s_of_prior, s_s_of_prior = func.load_summary_stats_mean_and_std()


# set up simulator
def simulator(theta):
    s_of_theta = model.model_sim(theta)

    return func.normalize_summary_stats(s_of_theta, m_s_of_prior, s_s_of_prior)


s_x_o = LotkaVolterra.calc_summary_stats(x_o.reshape(1, x_o.shape[0], x_o.shape[1]))
s_x_o = func.normalize_summary_stats(s_x_o, m_s_of_prior, s_s_of_prior)

# %%

simulator, prior = prepare_for_sbi(simulator, model.prior)

inference = SMCABC(simulator, prior, show_progress_bars=True)

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

x_o = s_x_o

# run inference w diff nbr of total sim

nbr_sim = [1000, 2000, 3000, 4000, 5000, 10000, 100000] #, 50000, 100000]

posteriors = []
run_time_save = 0

for n_tot in nbr_sim:
    print("Start: ABC-SMC, nbr sim = " + str(n_tot))

    start = time.time()

    posteriors.append(inference(x_o,
                                num_particles=1000,
                                num_initial_pop=1000,
                                num_simulations=n_tot,
                                epsilon_decay=0.05,
                                distance_based_decay=True))
    end = time.time()
    run_time = end - start

    if n_tot == 10000:
        run_time_save = run_time

    print("")
    print("Runtime:" + str(round(run_time, 2)))

print(len(posteriors))

print("- log prob(theta_true)")

log_probs = []

for i in range(len(posteriors)):
    posterior_samples = posteriors[i].sample_n(1000)
    post_gauss_approx = func.fit_gaussian_dist(posterior_samples)
    print(-post_gauss_approx.log_prob(theta_true))
    log_probs.append(-post_gauss_approx.log_prob(theta_true))

    # save samples
    np.savetxt('data/abcsmc_posterior_' + str(nbr_sim[i]) + '_' + id_job + '.csv',
               posterior_samples.detach().numpy(), delimiter=",")

# write new results
with open('results/abcsmc_' + id_job + '.txt', 'w') as f:
    f.write('%.4f\n' % run_time_save)
    for l in log_probs:
        f.write('%.4f\n' % l)
