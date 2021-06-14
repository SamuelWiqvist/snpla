# Imports
import sys
import torch
import os
import time
import numpy as np

# Initial set up
lunarc = int(sys.argv[1])
seed = int(sys.argv[2])

print("Input args:")
print("seed: " + str(seed))

if lunarc == 1:
    os.chdir('/home/samwiq/snpla/seq-posterior-approx-w-nf-dev/lotka_volterra')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/lotka_volterra')

sys.path.append('./')

print(os.getcwd())

id_job = str(seed)

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

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


prior_samples = model.prior.sample(sample_shape=(1,))
data_sets = simulator(prior_samples)

print(prior_samples)
print(data_sets)


print(prior_samples.shape)
print(data_sets.shape)

s_x_o = data_sets
x_o_batch_post = torch.zeros(batch_size_post, 9)

for i in range(batch_size_post):
    x_o_batch_post[i, :] = s_x_o

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

L = 4
M = L
K = 4

indications = torch.zeros(4)

post_samples = models_post[-1].sample(M, context=data_sets[0, :].reshape((1, 9)))
post_samples = post_samples.reshape((M, K))

for k in range(K):
    indications[k] = (post_samples[:, k] < prior_samples[0, k]).sum()

np.savetxt('sbc/ranks_snpla_' + id_job + '.csv', indications.numpy(), delimiter=",")

