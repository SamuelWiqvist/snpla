import sys
import os
import time

lunarc = int(sys.argv[1])
nbr_sim = int(sys.argv[2])

print("Input args:")
print("Nbr sim: " + str(nbr_sim))


# Set wd
print(os.getcwd())

# set the wd to the base folder for the project
if lunarc == 1:
    os.chdir('/home/samwiq/spa/seq-posterior-approx-w-nf-dev/lotka_volterra')
else:
    os.chdir('/home/samuel/Documents/projects/seq posterior approx w nf/seq posterior approx w nf dev/lotka_volterra')

sys.path.append('./')

print(os.getcwd())

import functions as func

x_o, model, theta_true = func.set_up_model()

start = time.time()

print("Start sim")

mu_s, std_s = func.gen_summary_stats_mean_and_std(model, nbr_sim)

end = time.time()
run_time = end - start
print(run_time)
