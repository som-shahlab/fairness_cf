outcomes='los mortality'
sensitive_variables='gender age race_eth'

for outcome in $outcomes
do
    for sensitive_variable in $sensitive_variables 
        do
            python train_cfvae.py --project_dir='/labs/shahlab/spfohl/fairness_MLHC/' \
                --outcome=$outcome \
                --sensitive_variable=$sensitive_variable \
                --config_path='config/defaults/cfvae/'$outcome'/'$sensitive_variable'/model_config.yaml' \
                --experiment_name='cfvae_default' \
                --trial_id=1
        done
done
