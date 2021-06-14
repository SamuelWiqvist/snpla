source /home/samuel/anaconda3/etc/profile.d/conda.sh
conda activate py374
python --version

# run simulations with seed = 1:MAX_SEED_VAL

echo Start runs

lambda_vals=(0.6 0.7 0.8 0.9)

for i in "${lambda_vals[@]}"; do

  python run_script_snpla.py 10 snl $i

done

echo Run done