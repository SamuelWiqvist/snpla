# Imports
import sys
import torch
import os
import numpy as np
import time
from torch.distributions.multivariate_normal import MultivariateNormal
from sbi.inference import SMCABC, prepare_for_sbi

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

id_job = str(dim) + '_' + str(seed) + '_' + str(seed_data)

print(os.getcwd())


# Load all utility functions for all methods
import mv_gaussian.low_dim_w_summary_stats.functions as func

# Set model and generate data

x_o, conj_model, analytical_posterior = func.set_up_model(seed)


# set up simulator
def simulator(theta):
    N_samples = theta.shape[0]

    x = torch.zeros(N_samples, conj_model.N, dim)

    for i in range(N_samples):
        model_tmp = MultivariateNormal(theta[i], conj_model.model.covariance_matrix)
        x[i, :, :] = model_tmp.rsample(sample_shape=(conj_model.N,))

    # return calc_summary_stats(x), theta #/math.sqrt(5) # div with std of prior to nomarlize data
    return func.calc_summary_stats(x)


# calc summary stats for obs data set
s_x_o = func.calc_summary_stats(x_o)


# check simulator and prior
simulator, prior = prepare_for_sbi(simulator, conj_model.prior)

inference = SMCABC(simulator, prior, show_progress_bars=False)

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

x_o = s_x_o

# run inference w diff nbr of total sim

print("ABC-SMC 2.5k")

posterior_2dot5k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=2500, epsilon_decay=0.8)

print("ABC-SMC 5k")

posterior_5k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=5000, epsilon_decay=0.8)

print("ABC-SMC 7.5k")

posterior_7dot5k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=7500, epsilon_decay=0.8)

print("ABC-SMC 10k")

start = time.time()

posterior_10k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=10000, epsilon_decay=0.8)

end = time.time()
run_time = end - start

print("")
print("Runtime:" + str(round(run_time, 2)))

print("ABC-SMC 100k")

posterior_100k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=100000, epsilon_decay=0.8)

print("ABC-SMC 1000k")

posterior_1000k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=1000000, epsilon_decay=0.8)

# sample from res posteriors

posterior_sample = posterior_2dot5k.sample((1000,))
kl_div_2dot5k = conj_model.kl_div(analytical_posterior, posterior_sample)

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/abcsmc_posterior_' + str(1) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")

posterior_sample = posterior_5k.sample((1000,))
kl_div_5k = conj_model.kl_div(analytical_posterior, posterior_sample)

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/abcsmc_posterior_' + str(2) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")

posterior_sample = posterior_7dot5k.sample((1000,))
kl_div_7dot5k = conj_model.kl_div(analytical_posterior, posterior_sample)

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/abcsmc_posterior_' + str(3) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")


start = time.time()

posterior_sample = posterior_10k.sample((1000,))
kl_div_10k = conj_model.kl_div(analytical_posterior, posterior_sample)

end = time.time()
run_time_inference = end - start

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/abcsmc_posterior_' + str(4) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")

posterior_sample = posterior_100k.sample((1000,))
kl_div_100k = conj_model.kl_div(analytical_posterior, posterior_sample)

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/abcsmc_posterior_' + str(100) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")


posterior_sample = posterior_1000k.sample((1000,))
kl_div_1000k = conj_model.kl_div(analytical_posterior, posterior_sample)

np.savetxt('mv_gaussian/low_dim_w_summary_stats/data/abcsmc_posterior_' + str(1000) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")

# save results
with open('mv_gaussian/low_dim_w_summary_stats/results/abcsmc_' + id_job + '.txt', 'w') as f:
    f.write('%.4f\n' % run_time)
    f.write('%.4f\n' % run_time_inference)
    f.write('%.4f\n' % kl_div_2dot5k)
    f.write('%.4f\n' % kl_div_5k)
    f.write('%.4f\n' % kl_div_7dot5k)
    f.write('%.4f\n' % kl_div_10k)
    f.write('%.4f\n' % kl_div_100k)
    f.write('%.4f\n' % kl_div_1000k)

