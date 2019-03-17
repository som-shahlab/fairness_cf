#!/bin/bash 
#SBATCH --job-name=train_cfvae_default
#SBATCH --output=slurm_out/train_cfvae_default%a.out
#SBATCH --error=slurm_out/train_cfvae_default%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:1

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
                --config_path='config/defaults/cfvae/'$outcome'/'$sensitive_variable'/model_config.yaml' \
                --experiment_name='cfvae_default' \
                --trial_id='0' \
                --save_checkpoints
        done
done
