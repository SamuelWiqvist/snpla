#!/bin/bash

MAX_SEED_VAL=10

for ((i=1;i<=$MAX_SEED_VAL;i++)); do

# check if file exists, if not create an empty file
#if [ ! -e ${FILES[$i]} ]; then
#  echo >> ${FILES[$i]}
#fi

FILE="job_snpla_${i}.sh"

mkdir -p lunarc_output

# create empty file
echo >> $FILE

cat > $FILE << EOF
#!/bin/sh

#SBATCH -A lu2020-2-7
#SBATCH -p lu

# time consumption HH:MM:SS
#SBATCH -t 5:00:00

#SBATCH -N 1
#SBATCH --tasks-per-node=1
# #SBATCH --exclusive

# name for script
#SBATCH -J snpla


# controll job outputs
#SBATCH -o lunarc_output/lunarc_output_snpla_%j.out
#SBATCH -e lunarc_output/lunarc_output_snpla_%j.err

# notification
#SBATCH --mail-user=samuel.wiqvist@matstat.lu.se
#SBATCH --mail-type=ALL

# load modules

ml load GCC/8.3.0
ml load CUDA/10.1.243
ml load OpenMPI/3.1.4
ml load PyTorch/1.6.0-Python-3.7.4

# run program
python /home/samwiq/snpla/'seq-posterior-approx-w-nf-dev'/'mv_gaussian'/low_dim_w_five_obs/run_script_snpla.py 1 2 $i 10 0 0
EOF


# run job
sbatch $FILE


done
