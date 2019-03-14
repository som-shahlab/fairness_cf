#!/bin/bash 
#SBATCH --job-name=tune_baseline
#SBATCH --output=slurm_out/tune_baseline_%a.out
#SBATCH --error=slurm_out/tune_baseline_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=64000
#SBATCH -c 8
#SBATCH --mail-user=spfohl@stanford.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-99

source activate /share/pi/nigam/spfohl/conda_envs/py_env

python train_baseline.py --project_dir='/share/pi/nigam/spfohl/fairness_cf/data' \
    --outcome='los' \
    --config_path='config/grid/baseline_v2/'$SLURM_ARRAY_TASK_ID'.yaml' \
    --experiment_name='baseline_grid_v2' \
    --trial_id=$SLURM_ARRAY_TASK_ID

python train_baseline.py --project_dir='/share/pi/nigam/spfohl/fairness_cf/data' \
    --outcome='mortality' \
    --config_path='config/grid/baseline_v2/'$SLURM_ARRAY_TASK_ID'.yaml' \
    --experiment_name='baseline_grid_v2' \
    --trial_id=$SLURM_ARRAY_TASK_ID
