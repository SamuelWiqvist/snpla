import os
import sys

seed_data = 7

lunarc = int(sys.argv[1])
nbr_params = int(sys.argv[2])
data_set = str(sys.argv[3])
lambda_val = float(sys.argv[4])
seed = int(sys.argv[5])
#1 10 snl $i 2

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
import sys

if lunarc == 1:
    sys.path.append('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev/algorithms')
else:
    sys.path.append(
        '/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/algorithms')

import snpla as snpla

nbr_samples = int(len(HodgkinHuxley.h.t_vec) * HodgkinHuxley.h.dt)

job = str(data_set) + "_" + str(nbr_params) + "_" + str(nbr_samples) + "_" + str(seed)  # + "extended"

if lambda_val > 0:
    job = job + "_" + str(lambda_val)

# Gen sbi data

model = HodgkinHuxley.HodgkinHuxley(data_set, nbr_params, "snpla")

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


flow_lik, flow_post = func.set_up_networks(model.prior.low,
                                           model.prior.high,
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
optimizer_post = torch.optim.Adam(flow_post.parameters(), lr=0.001, weight_decay=0.0)  # used def value before
decay_rate_post = 0.95  # was 0.95

s_x_o = torch.from_numpy(summary_stats_obs_w).to(dtype=torch.float32).reshape(1, 19)

nbr_rounds = 12

prob_prior_decay_rate = 0.8  # was 0.95

if lambda_val > 0:
    prob_prior_decay_rate = lambda_val

prob_prior = snpla.calc_prob_prior(nbr_rounds, prob_prior_decay_rate)

print(prob_prior)

nbr_lik = [2000 for _ in range(nbr_rounds)]  # [1000, 1000, 1000, 1000, 1000]  # , 2000, 2000]
nbr_epochs_lik = [100 for _ in range(nbr_rounds)]  # [100, 100, 100, 100, 100]
batch_size = 50
batch_size_post = 2000
nbr_post = [10000 for _ in range(nbr_rounds)]  # [10000, 10000, 10000, 10000, 10000]  # , 10000, 10000]
nbr_epochs_post = [50 for _ in range(nbr_rounds)]  # [50, 50, 50, 50, 50, 50]

x_o_batch_post = torch.zeros(batch_size_post, 19)

for i in range(batch_size_post):
    x_o_batch_post[i, :] = s_x_o

#print("----------------")
#print(model.prior.low)
#print(flow_post.sample(1000, context=s_x_o).min(dim=1))
#print("---")
#print(model.prior.high)
#print(flow_post.sample(1000, context=s_x_o).max(dim=1))

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dim_post = nbr_params
start = time.time()

models_lik, models_post = snpla.inference_snpla(flow_lik,
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

    if lambda_val > 0:

        np.savetxt('lambda_val/post_samples_snpla_' + str(i + 1) + "_" + job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")

    else:

        np.savetxt('data/post_samples_snpla_' + str(i + 1) + '_' + job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")

end = time.time()
run_time_inference = (end - start) / nbr_rounds

# Run likelihood tests

# calc true and pred normalized summary stats w samples form the final post

# calc true and pred normalized summary stats w samples form the final post
if lambda_val == 0:
    posterior_samples = models_post[-1].sample(1000, context=s_x_o)
    posterior_samples = posterior_samples.reshape((1000, 10))

    start = time.time()

    ture_normalized_summary_stats = simulator(posterior_samples)

    end = time.time()
    run_time_simulator = end - start

    start = time.time()

    pred_normalized_summary_stas = flow_lik.sample(1, context=posterior_samples)

    end = time.time()
    run_time_flow = end - start

    pred_normalized_summary_stas = pred_normalized_summary_stas.squeeze(1)

    ture_normalized_summary_stats = ture_normalized_summary_stats.detach().numpy()
    pred_normalized_summary_stas = pred_normalized_summary_stas.detach().numpy()

    ture_summary_stats = func.de_whiten(ture_normalized_summary_stats, whiteness_params)
    pred_summary_stas = func.de_whiten(pred_normalized_summary_stas, whiteness_params)

    np.savetxt('data/ture_summary_stats_post_samples_snpla_' + job + '.csv',
                   ture_summary_stats, delimiter=",")

    np.savetxt('data/pred_summary_stas_post_samples_snpla_' + job + '.csv',
                   pred_summary_stas, delimiter=",")

    # calc true and pred normalized summary stats w samples form the prior

    nbr_prior_samples = 1000

    prior_samples = model.prior.sample((nbr_prior_samples,))

    ture_normalized_summary_stats = simulator(prior_samples)
    pred_normalized_summary_stas = flow_lik.sample(1, context=prior_samples)
    pred_normalized_summary_stas = pred_normalized_summary_stas.squeeze(1)

    ture_normalized_summary_stats = ture_normalized_summary_stats.detach().numpy()
    pred_normalized_summary_stas = pred_normalized_summary_stas.detach().numpy()

    ture_summary_stats = func.de_whiten(ture_normalized_summary_stats, whiteness_params)
    pred_summary_stas = func.de_whiten(pred_normalized_summary_stas, whiteness_params)

    np.savetxt('data/ture_summary_stats_prio_samples_snpla_' + job + '.csv',
                   ture_summary_stats, delimiter=",")

    np.savetxt('data/pred_summary_stas_prior_samples_snpla_' + job + '.csv',
                   pred_summary_stas, delimiter=",")

# Write results
if lambda_val > 0:

    with open('lambda_val/snpla_' + job + '.txt', 'w') as f:
        f.write('%.6f\n' % prob_prior_decay_rate)
        for p in prob_prior:
            f.write('%.6f\n' % p)
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        f.write('%.4f\n' % run_time_simulator)
        f.write('%.4f\n' % run_time_flow)


else:

    with open('results/snpla_' + job + '.txt', 'w') as f:
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)

