outcomes='los mortality'

for outcome in $outcomes
do
    python train_baseline.py --project_dir='/labs/shahlab/spfohl/fairness_MLHC/' \
        --outcome=$outcome \
        --config_path='config/defaults/baseline/'$outcome'/model_config.yaml' \
        --experiment_name='baseline_default' \
        --trial_id=0
done