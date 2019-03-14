#!/bin/bash 
#SBATCH --job-name=tune_cfvae
#SBATCH --output=slurm_out/tune_cfvae_%a.out
#SBATCH --error=slurm_out/tune_cfvae_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-99


#rm ./slurm_out/*

source activate /share/pi/nigam/spfohl/conda_envs/py_env

outcomes='los mortality'
sensitive_variables='gender age race_eth'

for outcome in $outcomes
do
    for sensitive_variable in $sensitive_variables 
        do
            python train_cfvae.py --project_dir='/share/pi/nigam/spfohl/fairness_cf/data' \
                --outcome=$outcome \
                --sensitive_variable=$sensitive_variable\
                --config_path='config/grid/cfvae/'$sensitive_variable'/'$SLURM_ARRAY_TASK_ID'.yaml' \
                --experiment_name='cfvae_grid' \
                --trial_id=$SLURM_ARRAY_TASK_ID \
		--no_checkpoints
        done
done
