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
  parser.add_argument('--config_path',
    type=str,
    default='')
  parser.add_argument('--experiment_name',
    type=str,
    default='scratch')
  parser.add_argument('--trial_id',
    type=str,
    default='0')

  args = parser.parse_args()

  features_path = os.path.join(args.project_dir, 'features', str(0))
  label_path = os.path.join(args.project_dir, 'labels')
  config_path = os.path.join(args.project_dir, args.config_path)


  checkpoints_path = os.path.join(args.project_dir, 'checkpoints', args.experiment_name, args.outcome, args.sensitive_variable, args.trial_id)
  performance_path = os.path.join(args.project_dir, 'performance', args.experiment_name, args.outcome, args.sensitive_variable, args.trial_id)

  os.makedirs(checkpoints_path, exist_ok=True)
  os.makedirs(performance_path, exist_ok=True)

  time_str = str(time.time())

  features_dict = joblib.load(os.path.join(features_path, 'features.pkl'))
  master_label_dict = joblib.load(os.path.join(label_path, 'label_dict.pkl'))

  data_dict = {split: features_dict[split]['features'] for split in features_dict.keys()}
  label_dict = {split : master_label_dict[split][args.outcome] for split in master_label_dict.keys()}
  group_dict = {split : master_label_dict[split][args.sensitive_variable] for split in master_label_dict.keys()}

  with open(config_path, 'r') as fp:
      config_dict = yaml.load(fp)
      
  print(config_dict)

  if args.sensitive_variable == 'gender':
    data_dict = {k: v[group_dict[k] < 2] for k,v in data_dict.items()}
    label_dict = {k: v[group_dict[k] < 2] for k,v in label_dict.items()}
    group_dict = {k: v[group_dict[k] < 2] for k,v in group_dict.items()}

  model = CFVAEModel(config_dict)
  result = model.train(data_dict, label_dict, group_dict)
  result_eval = model.predict(data_dict, label_dict, group_dict, phases = ['val', 'test'])

  ## Save weights
  model.save_weights(os.path.join(checkpoints_path, '{}.chk'.format(time_str)))

  # Get results
  result_df_training = model.process_result_dict(result)
  result_df_eval = model.process_result_dict(result_eval[1])

  print(result_df_training)
  print(result_df_eval)

  ## Get performance by group
  result_df_training.to_csv(os.path.join(performance_path, '{}_training.csv'.format(time_str)), index = False)
  result_df_eval.to_csv(os.path.join(performance_path, '{}_eval.csv'.format(time_str)), index = False)
