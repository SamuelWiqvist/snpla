# Imports
import sys
import torch
import os
import numpy as np
import time
from torch.distributions.multivariate_normal import MultivariateNormal

# Initial set up
lunarc = int(sys.argv[1])
dim = int(sys.argv[2])
seed = int(sys.argv[3])
seed_data = int(sys.argv[4])
hp_tuning = int(sys.argv[5])  # if hp_tuning = 0, no hyper-param tuning, else hp_tuning for that sample of the hp
lambda_val = float(sys.argv[6])  # if hp_tuning = 0, no hyper-param tuning, else hp_tuning for that sample of the hp

print("Input args:")
print("Dim: " + str(dim))
print("seed: " + str(seed))
print("seed_data: " + str(seed_data))

# Set wd
print(os.getcwd())

# set the wd to the base folder for the project
if lunarc == 1:
    os.chdir('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev')

sys.path.append('./')

print(os.getcwd())

id_job = str(dim) + '_' + str(seed) + '_' + str(seed_data)

if hp_tuning > 0:
    id_job = id_job + "_" + str(hp_tuning)

if lambda_val > 0:
    id_job = id_job + "_" + str(lambda_val)

# Load all utility functions for all methods
import mv_gaussian.low_dim_w_learnable_summary_stats.functions as func
import algorithms.snpla as snpla

# Set model and generate data

x_o, conj_model, analytical_posterior = func.set_up_model(seed)

### Gen test data

## Generate test data

N_prior_pred_test = 1000
x_test, theta_test = func.run_model_sim(N_prior_pred_test, seed + 2, conj_model, analytical_posterior,
                                        conj_model.model.covariance_matrix, dim, True)
# Generate test data for obs data set

N_test_obs_data = 1000

x_test_obs_data = torch.zeros(N_test_obs_data, conj_model.N * conj_model.dim)
theta_test_obs_data = torch.zeros(N_test_obs_data, dim)

for i in range(N_test_obs_data):
    x_test_obs_data[i, :] = x_o.flatten()

    theta_test_obs_data[i, :] = conj_model.model.loc

# normalize??

### Set up networks for the likelihood model

# Base dist for posterior model
flow_lik, flow_post = func.set_up_networks()

hyper_params = [0.001, 0.002, 0.95, 0.7]  # lr_like, lr_post, gamma_post, gamma

if lambda_val > 0:
    hyper_params[-1] = lambda_val

if hp_tuning >= 2:
    hyper_params = func.sample_hp("snpla", hp_tuning)

optimizer_lik = torch.optim.Adam(flow_lik.parameters(), lr=hyper_params[0])
optimizer_post = torch.optim.Adam(flow_post.parameters(), lr=hyper_params[1])
decay_rate_post = hyper_params[2]  # no adaptation of Adam's base rate

# reshape ?? x_o.flatten().reshape(1, 10))

nbr_rounds = 10
prob_prior_decay_rate = hyper_params[3]
prob_prior = snpla.calc_prob_prior(nbr_rounds, prob_prior_decay_rate)

print(prob_prior)

#nbr_lik = [2000, 2000, 2000, 2000]
#nbr_epochs_lik = [25, 25, 25, 25]
#batch_size = 50
#batch_size_post = 50
#nbr_post = [10000, 10000, 10000, 10000]
#nbr_epochs_post = [25, 25, 25, 25]

nbr_lik = [2500 for _ in range(nbr_rounds)]  # [1000, 1000, 1000, 1000, 1000]  # , 2000, 2000]
nbr_epochs_lik = [75 for _ in range(nbr_rounds)]  # [100, 100, 100, 100, 100]
batch_size = 50
batch_size_post = 1000
nbr_post = [40000 for _ in range(nbr_rounds)]  # [10000, 10000, 10000, 10000, 10000]  # , 10000, 10000]
nbr_epochs_post = [75 for _ in range(nbr_rounds)]  # [50, 50, 50, 50, 50, 50]


x_o_batch_post = torch.zeros(batch_size_post, 10)

for i in range(batch_size_post):
    x_o_batch_post[i, :] = x_o.flatten().reshape(1, 10)

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start = time.time()

# TODO check prior and simulator
models_lik, models_post = snpla.inference_snpla(flow_lik,
                                            flow_post,
                                            conj_model.prior,
                                            conj_model.model_sim,
                                            optimizer_lik,
                                            optimizer_post,
                                            decay_rate_post,
                                            x_o.flatten().reshape(1, 10),
                                            x_o_batch_post,
                                            dim,
                                            prob_prior,
                                            nbr_lik,
                                            nbr_epochs_lik,
                                            nbr_post,
                                            nbr_epochs_post,
                                            batch_size,
                                            batch_size_post)

end = time.time()
run_time = end - start

print("")
print("Runtime:" + str(round(run_time, 2)))

kl_divs_trained = []
start = time.time()
torch.manual_seed(seed)

for i in range(nbr_rounds):
    print(i)
    posterior_sample = models_post[i].sample(1000, context=x_o.flatten().reshape(1, 10))
    posterior_sample = posterior_sample.reshape((1000, 2))

    kl_divs_trained.append(conj_model.kl_div(analytical_posterior, posterior_sample))

    if hp_tuning == 0 and lambda_val > 0:

        np.savetxt('mv_gaussian/low_dim_w_learnable_summary_stats/lambda_val/post_samples_snpla_' + str(i + 1) + "_" + id_job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")

    elif hp_tuning == 0:

        np.savetxt('mv_gaussian/low_dim_w_learnable_summary_stats/data/post_samples_snpla_' + str(i + 1) + "_" + id_job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")

    else:

        np.savetxt('mv_gaussian/low_dim_w_learnable_summary_stats/hp_tuning/post_samples_snpla_' + str(i + 1) + "_" + id_job + '.csv',
                       posterior_sample.detach().numpy(), delimiter=",")


end = time.time()
run_time_inference = (end - start) / nbr_rounds

if hp_tuning == 0 and lambda_val > 0:

    with open('mv_gaussian/low_dim_w_learnable_summary_stats/lambda_val/snpla_' + id_job + '.txt', 'w') as f:
        for h in hyper_params:
            f.write('%.6f\n' % h)
        for p in prob_prior:
            f.write('%.6f\n' % p)
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        for i in range(nbr_rounds):
            f.write('%.4f\n' % kl_divs_trained[i])


elif hp_tuning == 0:

    with open('mv_gaussian/low_dim_w_learnable_summary_stats/results/snpla_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        for i in range(nbr_rounds):
            f.write('%.4f\n' % kl_divs_trained[i])

else:

    with open('mv_gaussian/low_dim_w_learnable_summary_stats/hp_tuning/snpla_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % hp_tuning)
        for h in hyper_params:
            f.write('%.6f\n' % h)
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        for i in range(nbr_rounds):
            f.write('%.4f\n' % kl_divs_trained[i])

if hp_tuning == 0:

    ### Inference
    N_post_pred_test = 1000
    x_post_pred, theta_post_pred = func.run_model_sim(N_post_pred_test, seed + 3, conj_model, analytical_posterior,
                                                      conj_model.model.covariance_matrix, dim, False)

    torch.manual_seed(seed)
    x_untrained = flow_lik.sample(1, context=theta_test)
    x_res_untrained_obs = flow_lik.sample(1, context=theta_test_obs_data)
    x_res_post = flow_lik.sample(1, context=theta_post_pred)

    x_untrained = x_untrained.reshape(x_test.shape)
    x_res_untrained_obs = x_res_untrained_obs.reshape(x_test_obs_data.shape)
    x_res_post = x_res_post.reshape(x_post_pred.shape)

    np.savetxt('mv_gaussian/low_dim_w_learnable_summary_stats/data/data_recon_snpla_' + id_job + '.csv',
               x_res_untrained_obs.detach().numpy(), delimiter=",")

    np.savetxt('mv_gaussian/low_dim_w_learnable_summary_stats/data/data_recon_prior_snpla_' + id_job + '.csv',
               x_untrained.detach().numpy(), delimiter=",")

    np.savetxt('mv_gaussian/low_dim_w_learnable_summary_stats/data/data_recon_post_snpla_' + id_job + '.csv',
               x_res_post.detach().numpy(), delimiter=",")
