source /home/samuel/anaconda3/etc/profile.d/conda.sh
conda activate py374
python --version

# run simulations with seed = 1:MAX_SEED_VAL

echo Start runs

seeds=(1 2 3 4 5)

for i in "${seeds[@]}"; do

  python run_script_snpla.py 0 10 snl 0 $i

done

#echo SNPLA done

#for i in "${seeds[@]}"; do

#  python run_script_snpe_c.py 0 10 snl $i

#done

#echo SNPE-C done


echo Runs done