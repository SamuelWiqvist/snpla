#!/bin/sh

#SBATCH -A lu2020-2-7
#SBATCH -p lu

# time consumption HH:MM:SS
#SBATCH -t 4:00:00

#SBATCH -N 1
#SBATCH --tasks-per-node=1
# #SBATCH --exclusive

# name for script
#SBATCH -J snpe_c

# controll job outputs
#SBATCH -o lunarc_output/lunarc_output_snpe_c_%j.out
#SBATCH -e lunarc_output/lunarc_output_snpe_c_%j.err

# notification
#SBATCH --mail-user=samuel.wiqvist@matstat.lu.se
#SBATCH --mail-type=ALL

# load modules

ml load GCC/8.3.0
ml load CUDA/10.1.243
ml load OpenMPI/3.1.4
ml load PyTorch/1.6.0-Python-3.7.4

# run program
python /home/samwiq/spa/'seq-posterior-approx-w-nf-dev'/'mv_gaussian'/low_dim_w_summary_stats/run_script_snpe_c.py 1 2 6 10
