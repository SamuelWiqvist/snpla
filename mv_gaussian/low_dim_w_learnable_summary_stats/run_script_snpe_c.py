# Imports
import sys
import torch
import os
import numpy as np
import time
from torch.distributions.multivariate_normal import MultivariateNormal
from sbi.inference import SNPE_C, prepare_for_sbi

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

print(os.getcwd())

id_job = str(dim) + '_' + str(seed) + '_' + str(seed_data)

# Load all utility functions for all methods
import mv_gaussian.low_dim_w_learnable_summary_stats.functions as func

# Set model and generate data

x_o, conj_model, analytical_posterior = func.set_up_model(seed)


def simulator(theta):
    N_samples = theta.shape[0]

    x = torch.zeros(N_samples, conj_model.N, dim)

    for i in range(N_samples):
        model_tmp = MultivariateNormal(theta[i], conj_model.model.covariance_matrix)
        x[i, :, :] = model_tmp.rsample(sample_shape=(conj_model.N,))

    # return calc_summary_stats(x), theta #/math.sqrt(5) # div with std of prior to nomarlize data
    return func.flatten(x)


# check simulator and prior
simulator, prior = prepare_for_sbi(simulator, conj_model.prior)


# function that builds the network
def build_custom_post_net(batch_theta, batch_x):
    flow_lik, flow_post = func.set_up_networks()

    return flow_post


inference = SNPE_C(simulator, prior, density_estimator=build_custom_post_net)

start = time.time()

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_rounds = 10
x_o = x_o.flatten()
x_o = x_o.reshape(1, len(x_o))

posteriors = []
proposal = None

for i in range(num_rounds):
    posterior = inference(num_simulations=2500, proposal=proposal, max_num_epochs=100)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

end = time.time()
run_time = end - start

print("")
print("Runtime:" + str(round(run_time, 2)))

kl_divs_trained = []
start = time.time()

for i in range(num_rounds):
    print(i)
    posterior_sample = posteriors[i].sample((1000,), x=x_o)
    kl_divs_trained.append(conj_model.kl_div(analytical_posterior, posterior_sample))

    np.savetxt(
        'mv_gaussian/low_dim_w_learnable_summary_stats/data/post_samples_snpec_' + str(i + 1) + "_" + id_job + '.csv',
        posterior_sample.detach().numpy(), delimiter=",")

end = time.time()
run_time_inference = (end - start) / num_rounds

# Write results

with open('mv_gaussian/low_dim_w_learnable_summary_stats/results/snpec_' + id_job + '.txt', 'w') as f:
    f.write('%.4f\n' % run_time)
    f.write('%.4f\n' % run_time_inference)
    for i in range(num_rounds):
        f.write('%.4f\n' % kl_divs_trained[i])
