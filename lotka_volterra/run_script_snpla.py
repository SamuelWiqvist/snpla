# Imports
import sys
import torch
import os
import time
import numpy as np

# Initial set up
lunarc = int(sys.argv[1])
seed = int(sys.argv[2])
hp_tuning = int(sys.argv[3])  # if hp_tuning = 0, no hyper-param tuning, else hp_tuning for that sample of the hp
lambda_val = float(sys.argv[4])  # if hp_tuning = 0, no hyper-param tuning, else hp_tuning for that sample of the hp

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

if lambda_val > 0:
    id_job = id_job + "_" + str(lambda_val)

import LotkaVolterra
import functions as func  # Set model and generate data

if lunarc == 1:
    sys.path.append('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev/algorithms')
    import snpla as snpla
else:
    sys.path.append(
        '/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/algorithms')
    import snpla as snpla

# Set model and generate data

x_o, model, theta_true = func.set_up_model(method="snpla")
m_s_of_prior, s_s_of_prior = func.load_summary_stats_mean_and_std()

s_x_o = LotkaVolterra.calc_summary_stats(x_o.reshape(1, x_o.shape[0], x_o.shape[1]))
s_x_o = func.normalize_summary_stats(s_x_o, m_s_of_prior, s_s_of_prior)


# set up simulator
def simulator(theta):
    s_of_theta = model.model_sim(theta, True)

    return func.normalize_summary_stats(s_of_theta, m_s_of_prior, s_s_of_prior)


hyper_params = [0.001, 0.001, 0.9, 0.9]  # lr_like, lr_post, gamma_post, gamma

if lambda_val > 0:
    hyper_params[-1] = lambda_val

if hp_tuning >= 2:
    hyper_params = func.sample_hp("snpla", hp_tuning)

print(hyper_params)

flow_lik, flow_post = func.set_up_networks()
optimizer_lik = torch.optim.Adam(flow_lik.parameters(), lr=hyper_params[0])
optimizer_post = torch.optim.Adam(flow_post.parameters(), lr=hyper_params[1], weight_decay=0.0)  # used def value before
decay_rate_post = hyper_params[2]  # no adaptation of Adam's base rate

nbr_rounds = 5
prob_prior_decay_rate = hyper_params[3]
prob_prior = snpla.calc_prob_prior(nbr_rounds, prob_prior_decay_rate)

print(prob_prior)

nbr_lik = [1000, 1000, 1000, 1000, 1000]  # , 2000, 2000]
nbr_epochs_lik = [50, 50, 50, 50, 50]
batch_size = 25
batch_size_post = 2000
nbr_post = [10000, 10000, 10000, 10000, 10000]  # , 10000, 10000]
nbr_epochs_post = [25, 25, 25, 25, 25, 25]

x_o_batch_post = torch.zeros(batch_size_post, 9)

for i in range(batch_size_post):
    x_o_batch_post[i, :] = s_x_o

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dim_post = 4
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

print("")
print("Runtime:" + str(round(run_time, 2)))

log_probs = []
start = time.time()
torch.manual_seed(seed)

for i in range(nbr_rounds):
    print(i)

    posterior_sample = models_post[i].sample(1000, context=s_x_o)
    posterior_sample = posterior_sample.reshape((1000, 4))
    # log_probs.append(-models_post[i].log_prob(theta_true.reshape((1, 4)), context=s_x_o))
    post_gauss_approx = func.fit_gaussian_dist(posterior_sample.detach())  # to get correct prob
    log_probs.append(-post_gauss_approx.log_prob(theta_true))

    if hp_tuning == 0 and lambda_val > 0:

        np.savetxt('lambda_val/post_samples_snpla_' + str(i + 1) + "_" + id_job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")

    elif hp_tuning == 0:

        np.savetxt('data/post_samples_snpla_' + str(i + 1) + "_" + id_job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")

    else:

        np.savetxt('hp_tuning/post_samples_snpla_' + str(i + 1) + "_" + id_job + '.csv',
                   posterior_sample.detach().numpy(), delimiter=",")

end = time.time()
run_time_inference = (end - start) / nbr_rounds

if hp_tuning == 0 and lambda_val > 0:

    with open('lambda_val/snpla_' + id_job + '.txt', 'w') as f:
        for h in hyper_params:
            f.write('%.6f\n' % h)
        for p in prob_prior:
            f.write('%.6f\n' % p)
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        for i in range(nbr_rounds):
            f.write('%.4f\n' % log_probs[i])

elif hp_tuning == 0:

    with open('results/snpla_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        for i in range(nbr_rounds):
            f.write('%.4f\n' % log_probs[i])

else:

    with open('hp_tuning/snpla_' + id_job + '.txt', 'w') as f:
        f.write('%.4f\n' % hp_tuning)
        for h in hyper_params:
            f.write('%.6f\n' % h)
        f.write('%.4f\n' % run_time)
        f.write('%.4f\n' % run_time_inference)
        for i in range(nbr_rounds):
            f.write('%.4f\n' % log_probs[i])

if hp_tuning == 0:

    # test likelihood model

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

    # post pred samples
    torch.manual_seed(seed + 3)
    N_post_pred_test = 1000
    theta_post_pred = models_post[-1].sample(N_post_pred_test, context=s_x_o)
    theta_post_pred = theta_post_pred.reshape((N_post_pred_test, 4))

    theta_test = theta_test.reshape(N_prior_pred_test, 4)
    theta_test_obs_data = theta_test_obs_data.reshape(N_test_obs_data, 4)
    theta_post_pred = theta_post_pred.reshape(N_post_pred_test, 4)

    # gen samples from trained model
    torch.manual_seed(seed)
    x_prior = flow_lik.sample(1, context=theta_test)
    x_theta_true = flow_lik.sample(1, context=theta_test_obs_data)
    x_post = flow_lik.sample(1, context=theta_post_pred)

    x_prior = x_prior.reshape((N_prior_pred_test, 9))
    x_theta_true = x_theta_true.reshape((N_test_obs_data, 9))
    x_post = x_post.reshape((N_post_pred_test, 9))

    # Write results
    np.savetxt('data/data_recon_prior_snpla_' + id_job + '.csv', x_prior.detach().numpy(), delimiter=",")

    np.savetxt('data/data_recon_snpla_' + id_job + '.csv', x_theta_true.detach().numpy(), delimiter=",")

    np.savetxt('data/data_recon_post_snpla_' + id_job + '.csv', x_post.detach().numpy(), delimiter=",")

if hp_tuning == 0:  # SB rank test

    # TODO need the correlated stuff here

    N = 500
    L = 9
    M = L
    K = 4
    ranks = torch.zeros(N, K)

    prior_samples = model.prior.sample(sample_shape=(N,))
    data_sets = simulator(prior_samples)

    for n in range(N):

        indications = torch.zeros(4)

        post_samples = models_post[-1].sample(M, context=data_sets[n, :].reshape((1, 9)))
        post_samples = post_samples.reshape((M, K))

        for k in range(K):
            indications[k] = (post_samples[:, k] < prior_samples[n, k]).sum()

        ranks[n, :] = indications

    np.savetxt('sbc/ranks_snpla_' + id_job + '.csv',
               ranks.numpy(), delimiter=",")
