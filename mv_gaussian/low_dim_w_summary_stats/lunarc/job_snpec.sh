#!/bin/sh

# set input arguments

# need this since I use a LU project
#SBATCH -A snic2019-3-630

# use gpu nodes
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --exclusive

# #SBATCH --mem-per-cpu=5000

# time consumption HH:MM:SS
#SBATCH -t 1:00:00

# name for script
#SBATCH -J snpec

# controll job outputs
#SBATCH -o lunarc_output/outputs_snpec_%j.out
#SBATCH -e lunarc_output/errors_snpec_%j.err

# notification
#SBATCH --mail-user=samuel.wiqvist@matstat.lu.se
#SBATCH --mail-type=ALL

# load modules

ml load GCC/8.3.0
ml load CUDA/10.1.243
ml load OpenMPI/3.1.4
ml load PyTorch/1.6.0-Python-3.7.4

# run program
python /home/samwiq/spa/'seq-posterior-approx-w-nf-dev'/'mv_gaussian'/low_dim_w_summary_stats/run_script_snpe_c.py 1 2 $1 10
