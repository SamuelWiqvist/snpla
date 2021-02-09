import os

os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/hodgkin_huxley')

import torch
import HodgkinHuxley
import numpy as np
import functions as func
import time
import sys

sys.path.append(
    '/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/algorithms')

import spa as spa

import sys

print("Python version")
print(sys.version)

print("Version info.")
print(sys.version_info)

print(os.getcwd())

seed_data = 7

nbr_params = int(sys.argv[1])
data_set = str(sys.argv[2])
nbr_samples = int(len(HodgkinHuxley.h.t_vec) * HodgkinHuxley.h.dt)
job = str(data_set) + "_" + str(nbr_params) + "_" + str(nbr_samples)  # + "extended"

# Gen sbi data

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


# m_prior, std_prior = func.load_summary_stats_mean_and_std()

# w_sim_wrapper = lambda param: torch.as_tensor(func.whiten(simulator_wrapper(param), whiteness_params))


def simulator(params):
    N = params.shape[0]
    data = torch.zeros(params.shape[0], 19)

    for i in range(N):
        data[i, :] = torch.as_tensor(func.whiten(simulator_wrapper(params[i, :]), whiteness_params))

    return data


flow_lik, flow_post = func.set_up_networks(model.prior.base_dist.low,
                                           model.prior.base_dist.high,
                                           dim_post=model.nbr_params)

# setting for not exteded:
# decay_rate_post = 0.95
# prob_prior_decay_rate = 0.9
# 1000, 10000


# setting for exteded:
# decay_rate_post = 0.9
# prob_prior_decay_rate = 0.9
# 2000, 10000

optimizer_lik = torch.optim.Adam(flow_lik.parameters())
optimizer_post = torch.optim.Adam(flow_post.parameters(), lr=0.001)  # used def value before
decay_rate_post = 0.99  # was 0.95

s_x_o = torch.from_numpy(summary_stats_obs_w).to(dtype=torch.float32).reshape(1, 19)

nbr_rounds = 12
prob_prior_decay_rate = 0.7  # was 0.95
prob_prior = spa.calc_prob_prior(nbr_rounds, prob_prior_decay_rate)

print(prob_prior)

nbr_lik = [2000 for _ in range(nbr_rounds)]  # [1000, 1000, 1000, 1000, 1000]  # , 2000, 2000]
nbr_epochs_lik = [100 for _ in range(nbr_rounds)]  # [100, 100, 100, 100, 100]
batch_size = 50
batch_size_post = 50
nbr_post = [10000 for _ in range(nbr_rounds)]  # [10000, 10000, 10000, 10000, 10000]  # , 10000, 10000]
nbr_epochs_post = [50 for _ in range(nbr_rounds)]  # [50, 50, 50, 50, 50, 50]

x_o_batch_post = torch.zeros(batch_size_post, 19)

for i in range(batch_size_post):
    x_o_batch_post[i, :] = s_x_o

torch.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dim_post = nbr_params
start = time.time()

models_lik, models_post = spa.inference_spa(flow_lik,
                                            flow_post,
                                            model.prior,
                                            simulator,
                                            optimizer_lik,
                                            optimizer_post,
                                            decay_rate_post,
                                            s_x_o,
                                            x_o_batch_post,
                                            dim_post,
                                            prob_prior,
                                            nbr_lik,
                                            nbr_epochs_lik,
                                            nbr_post,
                                            nbr_epochs_post,
                                            batch_size,
                                            batch_size_post)

end = time.time()
run_time = end - start

start = time.time()

for i in range(nbr_rounds):
    posterior_sample = models_post[i].sample(1000, context=s_x_o)
    posterior_sample = posterior_sample.reshape((1000, 10))
    np.savetxt('data/post_samples_snpla_' + str(i + 1) + '_' + job + '.csv',
               posterior_sample.detach().numpy(), delimiter=",")

end = time.time()
run_time_inference = (end - start) / nbr_rounds

# Write results

with open('results/snpla_' + '_' + job + '.txt', 'w') as f:
    f.write('%.4f\n' % run_time)
    f.write('%.4f\n' % run_time_inference)
