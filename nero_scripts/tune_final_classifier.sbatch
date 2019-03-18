#!/bin/bash 
#SBATCH --job-name=tune_classifier
#SBATCH --output=slurm_out/tune_classifier%a.out
#SBATCH --error=slurm_out/tune_classifier%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-119

source activate /share/pi/nigam/spfohl/conda_envs/py_env

outcomes='los mortality'
sensitive_variables='gender age race_eth'

for outcome in $outcomes
do
    for sensitive_variable in $sensitive_variables 
        do
            python train_final_classifier.py --project_dir='/share/pi/nigam/spfohl/fairness_cf/data' \
                --outcome=$outcome \
                --sensitive_variable=$sensitive_variable \
                --config_path_vae='config/defaults/cfvae/'$outcome'/'$sensitive_variable'/model_config.yaml' \
                --config_path='config/grid/final_classifier/'$outcome'/'$sensitive_variable'/'$SLURM_ARRAY_TASK_ID'.yaml' \
                --experiment_name='final_classifier_grid' \
                --trial_id=$SLURM_ARRAY_TASK_ID \
		--no_checkpoints
        done
done
