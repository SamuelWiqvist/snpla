# Imports
import sys
import torch
import os
import numpy as np
import time
from sbi.inference import SMCABC, prepare_for_sbi

# Initial set up

seed_data = 7


lunarc = int(sys.argv[1])
nbr_params = int(sys.argv[2])
data_set = str(sys.argv[3])
seed = int(sys.argv[4])

# Set wd
print(os.getcwd())

# set the wd to the base folder for the project
if lunarc == 1:
    os.chdir('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev/lotka_volterra')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/lotka_volterra')

sys.path.append('./')

print(os.getcwd())

print("test")

# remove disp setting
if lunarc == 1 and 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']

if lunarc == 1:
    os.chdir('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev/hodgkin_huxley')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/hodgkin_huxley')

import torch
import HodgkinHuxley
import numpy as np
import functions as func
import time

nbr_samples = int(len(HodgkinHuxley.h.t_vec) * HodgkinHuxley.h.dt)
job = str(data_set) + "_" + str(nbr_params) + "_" + str(nbr_samples) + "_" + str(seed)

# Gen  data

model = HodgkinHuxley.HodgkinHuxley(data_set, nbr_params)

v_true, Iinj = model.simulator(model.log_theta_true, seed_data, True)

summary_stats_obs = model.calculate_summary_statistics(v_true)


# set up model simulator

def simulator_wrapper(params):
    # return tensor
    return model.calculate_summary_statistics(model.simulator(params, None))


# run pilot to calc mean and std of summary stats
whiteness_params = func.pilot_run(model, simulator_wrapper, summary_stats_obs)

summary_stats_obs_w = func.whiten(summary_stats_obs, whiteness_params)

w_sim_wrapper = lambda param: torch.as_tensor(func.whiten(simulator_wrapper(param), whiteness_params))

# run inference using SNPE

from sbi.inference import SNPE_C, prepare_for_sbi

print(model.prior)

simulator, prior = prepare_for_sbi(w_sim_wrapper, model.prior)


inference = SMCABC(simulator, prior, show_progress_bars=True)

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

x_o = torch.from_numpy(summary_stats_obs_w).to(dtype=torch.float32).reshape(1, 19)

# run inference w diff nbr of total sim

nbr_sim = 1000000

posteriors = []
run_time_save = 0

print("Start: ABC-SMC, nbr sim = " + str(nbr_sim))

start = time.time()

posteriors.append(inference(x_o,
                            num_particles=10000,
                            num_initial_pop=10000,
                            num_simulations=nbr_sim,
                            epsilon_decay=0.05,
                            distance_based_decay=True))

end = time.time()
run_time = end - start

print("Runtime:" + str(round(run_time, 2)))

print(len(posteriors))


for i in range(len(posteriors)):
    posterior_samples = posteriors[i].sample_n(1000)
    np.savetxt('data/post_samples_abcsmc_' + str(i + 1) + '_' + job + '.csv',
               posterior_samples.detach().numpy(), delimiter=",")

# write new results
with open('results/abcsmc_' + job + '.txt', 'w') as f:
    f.write('%.4f\n' % run_time)

