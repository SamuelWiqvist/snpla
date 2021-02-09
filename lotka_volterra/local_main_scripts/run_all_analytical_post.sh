# load correct python version
source /home/samuel/anaconda3/etc/profile.d/conda.sh
conda activate py374
python --version

# run simulations with seed = 1:MAX_SEED_VAL

echo Start runs

MAX_SEED_VAL=10 # we use seed = 1:MAX_SEED_VAL

seed=1
while [ $seed -le $MAX_SEED_VAL ]
do
  echo Run simulation with seed = $seed
  python /home/samuel/Documents/projects/'seq posterior approx w nf'/'seq posterior approx w nf dev'/lotka_volterra/run_script_analytical.py 0 $seed
  ((seed++))
done

echo All done