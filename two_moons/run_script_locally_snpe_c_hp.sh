source /home/samuel/anaconda3/etc/profile.d/conda.sh
conda activate py374
python --version

# run simulations with seed = 1:MAX_SEED_VAL

echo Start runs

# 1 2 11 10 $i

python run_script_snpe_c.py 0 2 11 10 1
python run_script_snpe_c.py 0 2 11 10 2
python run_script_snpe_c.py 0 2 11 10 3
python run_script_snpe_c.py 0 2 11 10 4
python run_script_snpe_c.py 0 2 11 10 5
python run_script_snpe_c.py 0 2 11 10 6
python run_script_snpe_c.py 0 2 11 10 7
python run_script_snpe_c.py 0 2 11 10 8
python run_script_snpe_c.py 0 2 11 10 9
python run_script_snpe_c.py 0 2 11 10 10

echo Run done