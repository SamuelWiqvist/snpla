#!/bin/sh

#SBATCH -A snic2019-3-630

# time consumption HH:MM:SS
#SBATCH -t 1:00:00

#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --exclusive

# name for script
#SBATCH -J snl

 
# controll job outputs
#SBATCH -o lunarc_output/lunarc_output_snl_%j.out
#SBATCH -e lunarc_output/lunarc_output_snl_%j.err

# notification
#SBATCH --mail-user=samuel.wiqvist@matstat.lu.se
#SBATCH --mail-type=ALL

# load modules

ml load GCC/8.3.0
ml load CUDA/10.1.243
ml load OpenMPI/3.1.4
ml load PyTorch/1.6.0-Python-3.7.4

# run program
python /home/samwiq/snpla/'seq-posterior-approx-w-nf-dev'/'two_moons'/run_script_smcabc.py 1 2 $i 10