# Imports
import sys
import torch
import os
from sbi.inference import SNLE_A, prepare_for_sbi
from sbi.utils import BoxUniform
import numpy as np
import time

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
    os.chdir('/home/samwiq/spa/seq-posterior-approx-w-nf-dev/two_moons')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/two_moons')

sys.path.append('./')

print(os.getcwd())

id_job = str(dim) + '_' + str(seed) + '_' + str(seed_data)

import functions as func

prior = BoxUniform(low=-2*torch.ones(2), high=2*torch.ones(2))
x_o, model = func.set_up_model(prior)


def simulator(theta):

    return torch.from_numpy(model.gen_single(theta)).to(dtype=torch.float32).reshape(1,2)


# check simulator and prior
simulator, prior = prepare_for_sbi(simulator, model.prior)


# function that builds the network
def build_custom_like_net(batch_theta, batch_x):
    flow_lik, flow_post = func.set_up_networks(seed)

    return flow_lik


inference = SNLE_A(simulator, prior, density_estimator=build_custom_like_net)

start = time.time()

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_rounds = 5
x_o = x_o.flatten()

posteriors = []
proposal = None

for i in range(num_rounds):
    print("Round: " + str(i))
    posterior = inference(num_simulations=2000, proposal=proposal, max_num_epochs=50)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

end = time.time()
run_time = end - start

print("")
print("Runtime:" + str(round(run_time, 2)))

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start = time.time()

for i in range(num_rounds):

    print(i)
    theta_trained = posteriors[i].sample((1000,), x=x_o)
    theta_trained = theta_trained.reshape((1000, 2))

    np.savetxt('data/snl_posterior_' + str(i+1) + "_" + id_job + '.csv',
               theta_trained.detach().numpy(), delimiter=",")

end = time.time()
run_time_inference = (end - start)/num_rounds

torch.manual_seed(seed)

theta_test = model.gen_posterior_samples(x_o, 1000)

x_trained = inference._posterior.net.sample(1, context=theta_test)
x_trained = x_trained.reshape((1000, 2))

np.savetxt('data/snl_post_pred_' + id_job + '.csv', x_trained.detach().numpy(), delimiter=",")


# save runtimes

with open('results/snl_' + id_job + '.txt', 'w') as f:

    f.write('%.4f\n' % run_time)
    f.write('%.4f\n' % run_time_inference)
