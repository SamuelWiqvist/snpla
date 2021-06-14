print("test")

import os
import sys

print("test")

seed_data = 7

# Gen sbi data
lunarc = int(sys.argv[1])
nbr_params = int(sys.argv[2])
data_set = str(sys.argv[3])
seed = int(sys.argv[4])

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
#job = str(data_set) + "_" + str(nbr_params) + "_" + str(nbr_samples)
job = str(data_set) + "_" + str(nbr_params) + "_" + str(nbr_samples) + "_" + str(seed)  # + "extended"
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

# run inference using SNL

from sbi.inference import SNLE_A, prepare_for_sbi

simulator, prior = prepare_for_sbi(w_sim_wrapper, model.prior)


def build_custom_lik_net(batch_theta, batch_x):
    flow_lik, flow_post = func.set_up_networks(model.prior.base_dist.low,
                                               model.prior.base_dist.high,
                                               dim_post=model.nbr_params)

    return flow_lik


print(summary_stats_obs)

print(summary_stats_obs_w)

inference = SNLE_A(simulator, prior, density_estimator=build_custom_lik_net)

start = time.time()

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

prior_samples = prior.sample(sample_shape=(1,))
data_sets = torch.zeros(1, 19)

for i in range(1):
    data_sets[i, :] = w_sim_wrapper(prior_samples[i, :])

print(prior_samples)
print(data_sets)

num_rounds = 12
x_o = data_sets

posteriors = []
proposal = None

for i in range(num_rounds):
    # lr = 0.001*math.exp(-0.95 * i)
    posterior = inference(num_simulations=2000, proposal=proposal, max_num_epochs=100)  # , learning_rate=0.001)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

end = time.time()
run_time = end - start

print("")
print("Runtime:" + str(round(run_time, 2)))

L = 5
K = nbr_params
M = L
Lprime = 50

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

for k in range(K):
    indications[k] = (post_samples[:, k] < prior_samples[0, k]).sum()

np.savetxt('sbc/ranks_snl_' + job + '.csv', indications.numpy(), delimiter=",")
