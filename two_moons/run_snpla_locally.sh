source /home/samuel/anaconda3/etc/profile.d/conda.sh
conda activate py374
python --version

# run simulations with seed = 1:MAX_SEED_VAL

echo Start runs

python run_script_snpla.py 0 2 1 10 0
python run_script_snpla.py 0 2 2 10 0
python run_script_snpla.py 0 2 3 10 0
python run_script_snpla.py 0 2 4 10 0
python run_script_snpla.py 0 2 5 10 0
python run_script_snpla.py 0 2 6 10 0
python run_script_snpla.py 0 2 7 10 0
python run_script_snpla.py 0 2 8 10 0
python run_script_snpla.py 0 2 9 10 0
python run_script_snpla.py 0 2 10 10 0

echo Run done
