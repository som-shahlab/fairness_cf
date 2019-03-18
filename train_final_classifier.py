import numpy as np
import os
import torch
import pandas as pd
import yaml
import time
import argparse

from sklearn.externals import joblib

from pytorch_utils.cfvae_models import CFVAEModel
import pytorch_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir',
                      type=str,
                      default='/labs/shahlab/spfohl/fairness_MLHC/')
    parser.add_argument('--outcome',
                      type = str,
                      default = 'los')
    parser.add_argument('--sensitive_variable',
                      type = str,
                      default = 'age')
    parser.add_argument('--config_path_vae',
    type=str,
    default='')
    parser.add_argument('--checkpoints_path_vae',
    type=str,
    default='')
    parser.add_argument('--experiment_name',
    type=str,
    default='scratch')
    parser.add_argument('--trial_id',
    type=str,
    default='0')
    parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true')
    parser.add_argument('--no_checkpoints', dest='save_checkpoints', action='store_false')
    parser.set_defaults(save_checkpoints=True)

    args = parser.parse_args()
    features_path = os.path.join(args.project_dir, 'features', str(0), '{}_excluded'.format(args.sensitive_variable))
    label_path = os.path.join(args.project_dir, 'labels')

    ## Config and checkpoint paths for loading VAE parameters
    if args.config_path_vae == '':
        config_path_vae = os.path.join(args.project_dir, 'config', 'defaults', 'cfvae', args.outcome, args.sensitive_variable, 'model_config.yaml')
    else:
        config_path_vae = os.path.join(args.project_dir, args.config_path_vae)

    if args.checkpoints_path_vae == '':
        checkpoints_path_vae = os.path.join(args.project_dir, 'checkpoints', 'cfvae_default', args.outcome, args.sensitive_variable, str(args.trial_id))
    else:
        checkpoints_path_vae = os.path.join(args.project_dir, args.checkpoints_path_vae)

    if args.save_checkpoints:
        checkpoints_path = os.path.join(args.project_dir, 'checkpoints', args.experiment_name, args.outcome, args.sensitive_variable, args.trial_id)
        os.makedirs(checkpoints_path, exist_ok=True)

    performance_path = os.path.join(args.project_dir, 'performance', args.experiment_name, args.outcome, args.sensitive_variable, args.trial_id)
    os.makedirs(performance_path, exist_ok=True)
    time_str = str(time.time())

    features_dict = joblib.load(os.path.join(features_path, 'features.pkl'))
    master_label_dict = joblib.load(os.path.join(label_path, 'label_dict.pkl'))

    data_dict = {split: features_dict[split]['features'] for split in features_dict.keys()}
    label_dict = {split : master_label_dict[split][args.outcome] for split in master_label_dict.keys()}
    group_dict = {split : master_label_dict[split][args.sensitive_variable] for split in master_label_dict.keys()}

    with open(config_path_vae, 'r') as fp:
        config_dict_vae = yaml.load(fp)
      
    print(config_dict_vae)

    if args.sensitive_variable == 'gender':
        data_dict = {k: v[group_dict[k] < 2] for k,v in data_dict.items()}
        label_dict = {k: v[group_dict[k] < 2] for k,v in label_dict.items()}
        group_dict = {k: v[group_dict[k] < 2] for k,v in group_dict.items()}

    config_dict_final_classifier = {
        'lr_final_classifier' : 1e-3,
        'lambda_final_classifier_cf' : 0e0,
        'lambda_clp' : 5e0,
        'lambda_clp_entropy' : 0e0,
        'num_epochs' : 10,
        'weighted' : True,
        'num_samples_eval': 1
    }

    config_dict_vae.update(config_dict_final_classifier)
    model = CFVAEModel(config_dict_vae)
    model.load_weights(os.path.join(checkpoints_path_vae, os.listdir(checkpoints_path_vae)[0]))
    result = model.train_final_classifier(data_dict, label_dict, group_dict)
    result_eval_CLP = model.predict_final_classifier_CLP(data_dict, label_dict, group_dict, phases = ['val', 'test'])

    result_df_training = model.process_result_dict(result)
    result_df_eval = model.process_result_dict(result_eval_CLP[1])

    cf_dict = result_eval_CLP[2]
    cf_df = pd.concat({key: pd.DataFrame(cf_dict[key]) for key in cf_dict.keys()}).rename_axis(index = ['phase', 'id']).reset_index(0)
    cf_df = cf_df.assign(pred_diff = lambda x: x.pred_prob_cf - x.pred_prob_factual)

    ## Get performance by group
    sensitive_variables = ['age', 'gender', 'race_eth']
    data_dict_by_group = {sensitive_variable: {} for sensitive_variable in sensitive_variables}
    label_dict_by_group = {sensitive_variable: {} for sensitive_variable in sensitive_variables}
    group_dict_by_group = {sensitive_variable: {} for sensitive_variable in sensitive_variables}
    for sensitive_variable in sensitive_variables:
        groups = np.unique(master_label_dict['train'][sensitive_variable])
        for group in groups:
            data_dict_by_group[sensitive_variable][group] = {split: 
                                         data_dict[split][master_label_dict[split][sensitive_variable] == group]
                                         for split in data_dict.keys()
                                        }
            label_dict_by_group[sensitive_variable][group] = {split: 
                                         label_dict[split][master_label_dict[split][sensitive_variable] == group]
                                         for split in data_dict.keys()
                                        }
            group_dict_by_group[sensitive_variable][group] = {split: 
                                         group_dict[split][master_label_dict[split][sensitive_variable] == group]
                                         for split in data_dict.keys()
                                        }

    result_df_by_group = pd.concat({sensitive_variable: 
                              pd.concat({
                                  group: model.process_result_dict(model.predict_final_classifier(data_dict_by_group[sensitive_variable][group],
                                                                                 label_dict_by_group[sensitive_variable][group], 
                                                                                 group_dict_by_group[sensitive_variable][group],
                                                      phases = ['val', 'test'])[1])
                                  for group in data_dict_by_group[sensitive_variable].keys()
                              })
                              for sensitive_variable in data_dict_by_group.keys()
                             })
    result_df_by_group.index = result_df_by_group.index.set_names(['sensitive_variable', 'group', 'index'])
    result_df_by_group = result_df_by_group.reset_index(level = [0, 1])

    result_df_training.to_csv(os.path.join(performance_path, '{}_training.csv'.format(time_str)), index = False)
    result_df_eval.to_csv(os.path.join(performance_path, '{}_eval.csv'.format(time_str)), index = False)
    result_df_by_group.to_csv(os.path.join(performance_path, '{}_by_group.csv'.format(time_str)), index = False)
    cf_df.to_csv(os.path.join(performance_path, '{}_cf_df.csv'.format(time_str)), index = False)