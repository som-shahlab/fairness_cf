import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from .layers import *
from .datasets import *

class model_CLP(TorchModel):
    
    def init_loaders(self, data_dict, data_dict_cf, label_dict):
        """
        Creates data loaders from inputs
        """
        dataset_dict = {key: TensorDataset(torch.FloatTensor(data_dict[key]), 
                                           torch.FloatTensor(data_dict_cf[key]),
                                           torch.LongTensor(label_dict[key])) 
                            for key in data_dict.keys()
                       }
        loaders_dict = {key: DataLoader(dataset_dict[key], 
                                        batch_size = self.config_dict['batch_size']) 
                            for key in data_dict.keys()
                       }
        return loaders_dict
    
    def init_loss_dict(self, metrics = ['loss', 'classification', 'clp'], phases = ['train', 'val']):
        return super().init_loss_dict(metrics = metrics, phases = phases)
    
    def train(self, data_dict, data_dict_cf, label_dict):
        loaders = self.init_loaders(data_dict, data_dict_cf, label_dict)
        best_performance = 1e18
        loss_dict = self.init_loss_dict()
        performance_dict = self.init_performance_dict()
        
        for epoch in range(self.config_dict['num_epochs']):
            print('Epoch {}/{}'.format(epoch, self.config_dict['num_epochs'] - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                self.model.train(phase == 'train')
                running_loss_dict = self.init_running_loss_dict(list(loss_dict[phase].keys()))
                output_dict = self.init_output_dict()
                i = 0
                for the_data in loaders[phase]:
                    i += 1
                    batch_loss_dict = {}
                    inputs, inputs_cf, labels = self.transform_batch(the_data)
                    self.optimizer.zero_grad()
                    
                    outputs = self.model(inputs)
                    
                    output_dict = self.update_output_dict(output_dict, outputs, labels)
                    
                    outputs_cf = self.model(inputs_cf)
                    
                    batch_loss_dict['classification'] = self.criterion(outputs, labels)
                    batch_loss_dict['clp'] = ((outputs - outputs_cf) ** 2).mean()
                    
                    batch_loss_dict['loss'] = batch_loss_dict['classification'] + \
                                                self.config_dict['lambda_cf'] * batch_loss_dict['clp']
                    if phase == 'train':
                        batch_loss_dict['loss'].backward()
                        self.optimizer.step()
                        
                    for key in batch_loss_dict.keys():
                        running_loss_dict[key] += batch_loss_dict[key].item()

                # Compute epoch losses and update loss dict
                epoch_loss_dict = {key: running_loss_dict[key] / i for key in running_loss_dict.keys()}
                loss_dict[phase] = self.update_metric_dict(loss_dict[phase], epoch_loss_dict)
                
                # Compute epoch performance and update performance dict
                epoch_statistics = self.compute_epoch_statistics(output_dict)
                performance_dict[phase] = self.update_metric_dict(performance_dict[phase], epoch_statistics)
                
                print('Phase: {}:'.format(phase))
                self.print_metric_dict(epoch_loss_dict)
                self.print_metric_dict(epoch_statistics)
                
                if phase == 'val':
                    best_model_condition = epoch_loss_dict['loss'] < best_performance
                    if best_model_condition:
                        print('Best model updated')
                        best_performance = epoch_loss_dict['loss']
                        best_model_wts = copy.deepcopy(self.model.state_dict())

        print('Best val performance: {:4f}'.format(best_performance))
        self.model.load_state_dict(best_model_wts)
        result_dict = {phase: {**performance_dict[phase], **loss_dict[phase]} for phase in performance_dict.keys()}
        return result_dict

class model_CLP_conditional(TorchModel):
    
    def init_loaders(self, data_dict, data_dict_cf, label_dict, label_dict_cf):
        """
        Creates data loaders from inputs
        """
        dataset_dict = {key: TensorDataset(torch.FloatTensor(data_dict[key]), 
                                           torch.FloatTensor(data_dict_cf[key]),
                                           torch.LongTensor(label_dict[key]),
                                           torch.LongTensor(label_dict_cf[key])
                                          ) 
                            for key in data_dict.keys()
                       }
        loaders_dict = {key: DataLoader(dataset_dict[key], 
                                        batch_size = self.config_dict['batch_size']) 
                            for key in data_dict.keys()
                       }
        return loaders_dict
    
    def init_loss_dict(self, metrics = ['loss', 'classification', 'clp', 'classification_cf'], 
                       phases = ['train', 'val']):
        return super().init_loss_dict(metrics = metrics, phases = phases)
    
    def train(self, data_dict, data_dict_cf, label_dict, label_dict_cf):
        loaders = self.init_loaders(data_dict = data_dict, data_dict_cf = data_dict_cf, 
                                    label_dict = label_dict, label_dict_cf = label_dict_cf)
        best_performance = 1e18
        loss_dict = self.init_loss_dict()
        performance_dict = self.init_performance_dict()
        performance_dict_cf = self.init_performance_dict()
        
        for epoch in range(self.config_dict['num_epochs']):
            print('Epoch {}/{}'.format(epoch, self.config_dict['num_epochs'] - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                self.model.train(phase == 'train')
                running_loss_dict = self.init_running_loss_dict(list(loss_dict[phase].keys()))
                output_dict = self.init_output_dict()
                output_dict_cf = self.init_output_dict()
                i = 0
                for the_data in loaders[phase]:
                    i += 1
                    batch_loss_dict = {}
                    inputs, inputs_cf, labels, labels_cf = self.transform_batch(the_data)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    output_dict = self.update_output_dict(output_dict, outputs, labels)
                    
                    outputs_cf = self.model(inputs_cf)
                    output_dict_cf = self.update_output_dict(output_dict_cf, outputs_cf, labels_cf)
                    
                    ## Classification losses
                    batch_loss_dict['classification'] = self.criterion(outputs, labels)
                    batch_loss_dict['classification_cf'] = self.criterion(outputs_cf, labels_cf)
                    
                    ## Conditional CLP
                    mask = labels == labels_cf
                    
                    batch_loss_dict['clp'] = ((outputs[mask] - outputs_cf[mask]) ** 2).mean()
                    
                    batch_loss_dict['loss'] = (batch_loss_dict['classification'] + 
                                self.config_dict['lambda_cls_cf'] * batch_loss_dict['classification_cf'] + 
                                self.config_dict['lambda_clp'] * batch_loss_dict['clp']
                           )
                    if phase == 'train':
                        batch_loss_dict['loss'].backward()
                        self.optimizer.step()
                    
                    for key in batch_loss_dict.keys():
                        running_loss_dict[key] += batch_loss_dict[key].item()
                    
                
                # Compute epoch losses and update loss dict
                epoch_loss_dict = {key: running_loss_dict[key] / i for key in running_loss_dict.keys()}
                loss_dict[phase] = self.update_metric_dict(loss_dict[phase], epoch_loss_dict)
                
                # Compute epoch performance and update performance dict
                epoch_statistics = self.compute_epoch_statistics(output_dict)
                performance_dict[phase] = self.update_metric_dict(performance_dict[phase], epoch_statistics)
                
                epoch_statistics_cf = self.compute_epoch_statistics(output_dict_cf)
                performance_dict_cf[phase] = self.update_metric_dict(performance_dict_cf[phase], epoch_statistics_cf)
                
                print('Phase: {}:'.format(phase))
                self.print_metric_dict(epoch_loss_dict)
                print('Factual')
                self.print_metric_dict(epoch_statistics)
                print('Counterfactual')
                self.print_metric_dict(epoch_statistics_cf)
            
                if phase == 'val':
                    best_model_condition = epoch_loss_dict['loss'] < best_performance
                    if best_model_condition:
                        print('Best model updated')
                        best_performance = epoch_loss_dict['loss']
                        best_model_wts = copy.deepcopy(self.model.state_dict())

        print('Best val performance: {:4f}'.format(best_performance))    
        self.model.load_state_dict(best_model_wts)
        return loss_dict, performance_dict, performance_dict_cf