# Imports
import sys
import torch
import os
import numpy as np
import time
from sbi.inference import SMCABC, prepare_for_sbi
from sbi.utils import BoxUniform

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

inference = SMCABC(simulator, prior, show_progress_bars=False)

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

x_o = x_o.flatten()

# run inference w diff nbr of total sim

print("ABC-SMC 2k")

posterior_2k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=2000, epsilon_decay=0.8)

print("ABC-SMC 4k")

posterior_4k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=4000, epsilon_decay=0.8)

print("ABC-SMC 6k")

posterior_6k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=6000, epsilon_decay=0.8)

print("ABC-SMC 8k")

posterior_8k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=8000, epsilon_decay=0.8)


print("ABC-SMC 10k")

start = time.time()

posterior_10k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=10000, epsilon_decay=0.8)

end = time.time()
run_time = end - start

print("")
print("Runtime:" + str(round(run_time, 2)))

print("ABC-SMC 100k")

posterior_100k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=100000, epsilon_decay=0.8)

print("ABC-SMC 1000k")

posterior_1000k = inference(x_o, num_particles=50, num_initial_pop=100, num_simulations=1000000, epsilon_decay=0.8)

# sample from res posteriors

posterior_sample = posterior_2k.sample((1000,))

np.savetxt('data/abcsmc_posterior_' + str(1) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")

posterior_sample = posterior_4k.sample((1000,))

np.savetxt('data/abcsmc_posterior_' + str(2) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")

posterior_sample = posterior_6k.sample((1000,))

np.savetxt('data/abcsmc_posterior_' + str(3) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")


posterior_sample = posterior_8k.sample((1000,))

np.savetxt('data/abcsmc_posterior_' + str(4) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")

start = time.time()

posterior_sample = posterior_10k.sample((1000,))

end = time.time()
run_time_inference = end - start

np.savetxt('data/abcsmc_posterior_' + str(5) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")

posterior_sample = posterior_100k.sample((1000,))

np.savetxt('data/abcsmc_posterior_' + str(100) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")

posterior_sample = posterior_1000k.sample((1000,))

np.savetxt('data/abcsmc_posterior_' + str(1000) + "_" + id_job + '.csv',
           posterior_sample.detach().numpy(), delimiter=",")

# save results
with open('results/abcsmc_' + id_job + '.txt', 'w') as f:
    f.write('%.4f\n' % run_time)
    f.write('%.4f\n' % run_time_inference)

