import numpy as np
import os
import torch
import pandas as pd
import yaml
import time
import argparse

from sklearn.externals import joblib

from pytorch_utils.datasets import ArrayDataset
from pytorch_utils.models import FeedforwardNetModel
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
    default='config/defaults/baseline.yaml')
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

  checkpoints_path = os.path.join(args.project_dir, 'checkpoints', args.experiment_name, args.outcome, args.trial_id)
  performance_path = os.path.join(args.project_dir, 'performance', args.experiment_name, args.outcome, args.trial_id)

  os.makedirs(checkpoints_path, exist_ok=True)
  os.makedirs(performance_path, exist_ok=True)

  time_str = str(time.time())

  features_dict = joblib.load(os.path.join(features_path, 'features.pkl'))
  master_label_dict = joblib.load(os.path.join(label_path, 'label_dict.pkl'))

  data_dict = {split: features_dict[split]['features'] for split in features_dict.keys()}
  label_dict = {split : master_label_dict[split][args.outcome] for split in master_label_dict.keys()}

  with open(config_path, 'r') as fp:
      config_dict = yaml.load(fp)
      
  # config_dict['num_epochs'] = 3 # For testing

  ## A more complex network
  # config_dict = {
  #     'input_dim' : data_dict['train'].shape[1],
  #     'lr' : 1e-5,
  #     'num_epochs' : 20,
  #     'batch_size' : 256,
  #     'hidden_dim' : 128,
  #     'num_hidden' : 1,
  #     'output_dim' : 2,
  #     'drop_prob' : 0.5,
  #     'normalize' : True,
  #     'iters_per_epoch' : 100,
  #     'gamma' : 0.99,
  #     'resnet' : True,
  #     'sparse' : True,
  #     'sparse_mode' : 'binary'
  # }
  print(config_dict)

  model = FeedforwardNetModel(config_dict)
  result = model.train(data_dict, label_dict)
  result_eval = model.predict(data_dict, label_dict, phases = ['val', 'test'])

  ## Save weights
  model.save_weights(os.path.join(checkpoints_path, '{}.chk'.format(time_str)))

  # Get results
  result_df_training = model.process_result_dict(result)
  result_df_eval = model.process_result_dict(result_eval[1])

  print(result_df_training)
  print(result_df_eval)

  ## Get performance by group
  sensitive_variables = ['race_eth', 'gender', 'age']
  data_dict_by_group = {sensitive_variable: {} for sensitive_variable in sensitive_variables}
  label_dict_by_group = {sensitive_variable: {} for sensitive_variable in sensitive_variables}
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
  result_df_by_group = pd.concat({sensitive_variable: 
                              pd.concat({
                                  group: model.process_result_dict(model.predict(data_dict_by_group[sensitive_variable][group],
                                                      label_dict_by_group[sensitive_variable][group],
                                                      phases = ['val', 'test'])[1])
                                  for group in data_dict_by_group[sensitive_variable].keys()
                              })
                              for sensitive_variable in data_dict_by_group.keys()
                             })
  result_df_by_group.index = result_df_by_group.index.set_names(['sensitive_variable', 'group', 'index'])
  result_df_by_group = result_df_by_group.reset_index(level = [0, 1])
  result_df_by_group.head()

  print(result_df_by_group)

  result_df_training.to_csv(os.path.join(performance_path, '{}_training'.format(time_str)), index = False)
  result_df_eval.to_csv(os.path.join(performance_path, '{}_eval'.format(time_str)), index = False)
  result_df_by_group.to_csv(os.path.join(performance_path, '{}_by_group'.format(time_str)), index = False)
