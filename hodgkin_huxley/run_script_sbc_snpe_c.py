import os
import sys

seed_data = 7

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


def build_custom_post_net(batch_theta, batch_x):
    flow_lik, flow_post = func.set_up_networks(model.prior.base_dist.low,
                                               model.prior.base_dist.high,
                                               dim_post=model.nbr_params)

    return flow_post


print(summary_stats_obs)

print(summary_stats_obs_w)

inference = SNPE_C(simulator, prior, density_estimator=build_custom_post_net)

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

print("----------------")
flow_post = build_custom_post_net(3, 4)
print(model.prior.base_dist.low)
print(flow_post.sample(1000, context=x_o).min(dim=1))
print("---")
print(model.prior.base_dist.high)
print(flow_post.sample(1000, context=x_o).max(dim=1))

posteriors = []
proposal = None

for i in range(num_rounds):
    print(i)
    # lr = 0.001*math.exp(-0.95 * i)
    posterior = inference(num_simulations=2000, proposal=proposal, max_num_epochs=100)  # , learning_rate=lr)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

end = time.time()
run_time = end - start

L = 5
M = L
K = nbr_params

indications = torch.zeros(K)

post_samples = posteriors[-1].net.sample(M, context=x_o.reshape((1, 19)))
post_samples = post_samples.reshape((M, K))

print(post_samples.shape)

print(prior_samples.shape)

for k in range(K):
    indications[k] = (post_samples[:, k] < prior_samples[0, k]).sum()

np.savetxt('sbc/ranks_snpec_' + job + '.csv', indications.numpy(), delimiter=",")
