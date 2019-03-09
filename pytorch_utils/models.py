import pandas as pd
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from .layers import *
from .datasets import *

class TorchModel:
    """
    Pytorch Model. 
    Default is logistic regression. Subclass and override init_model() for custom usage
    """
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()
        self.model.apply(self.weights_init)
        self.model.to(self.device)
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.criterion = self.init_loss()

    def init_datasets(self, data_dict, label_dict):
        return {key: TensorDataset(torch.FloatTensor(data_dict[key]), 
                                           torch.LongTensor(label_dict[key])) 
                            for key in data_dict.keys()
                }

    def init_loaders(self, data_dict, label_dict):
        """
        Creates data loaders from inputs
        """
        dataset_dict = self.init_datasets(data_dict, label_dict)

        # Determine which collate_fn is appropriate
        if hasattr(dataset_dict['train'], 'collate_fn') and callable(getattr(dataset_dict['train'], 'collate_fn')):
            collate_fn = dataset_dict['train'].collate_fn
        else:
            collate_fn = default_collate

        # Switch to handle random iteration versus looping the whole dataset
        if self.config_dict.get('iters_per_epoch') is not None:
            num_samples = self.config_dict['iters_per_epoch']*self.config_dict['batch_size']
            loaders_dict = {}
            for key in dataset_dict.keys():
                if key == 'train':
                    loaders_dict[key] = DataLoader(dataset_dict[key],
                                        batch_sampler = BatchSampler(
                                                            RandomSampler(dataset_dict[key],
                                                                replacement = True,
                                                                num_samples = num_samples
                                                                ),
                                                            batch_size = self.config_dict['batch_size'],
                                                            drop_last = False, 
                                                        ),
                                        collate_fn = collate_fn
                                ) 
                else:
                    loaders_dict[key] = DataLoader(dataset_dict[key], 
                                                    batch_size = self.config_dict['batch_size'],
                                                    collate_fn = collate_fn)
        else:
            loaders_dict = {key: DataLoader(dataset_dict[key], 
                                        batch_size = self.config_dict['batch_size'],
                                        collate_fn = collate_fn) 
                            for key in data_dict.keys()
                        }

        return loaders_dict
    
    def init_loaders_predict(self, data_dict, label_dict):
        """
        Creates data loaders from inputs - for use at prediction time
        """
        return self.init_loaders(data_dict, label_dict)
    
    def init_metric_dict(self, metrics = [''], phases = ['train', 'val']):
        """
        Initialize a dict of metrics
        """
        metric_dict = {phase : {metric: [] for metric in metrics} for phase in phases}
        return metric_dict
    
    def update_metric_dict(self, metric_dict, update_dict):
        """
        Updates a metric dict with metrics from an epoch
        """
        for key in update_dict.keys():
            metric_dict[key].append(update_dict[key])
        return metric_dict    
    
    def init_loss_dict(self, metrics = ['loss'], phases = ['train', 'val']):
        """
        Initialize
        """
        return self.init_metric_dict(metrics = metrics, phases = phases)
    
    def init_running_loss_dict(self, metrics):
        return {key : 0.0 for key in metrics}
    
    def init_performance_dict(self, metrics = ['auc', 'auprc', 'brier'], phases = ['train', 'val']):
        return self.init_metric_dict(metrics = metrics, phases = phases)
    
    def init_output_dict(self):
        return {'outputs' : torch.FloatTensor(),
                'pred_probs' : torch.FloatTensor(),
                'labels' : torch.LongTensor()
               }
    
    def update_output_dict(self, output_dict, outputs, labels):
        """
        Update an output_dict
        """
        pred_probs = F.softmax(outputs, dim = 1)
        
        output_dict['outputs'] = torch.cat((output_dict['outputs'], outputs.detach().cpu()))
        output_dict['pred_probs'] = torch.cat((output_dict['pred_probs'], pred_probs.detach().cpu()))
        output_dict['labels'] = torch.cat((output_dict['labels'], labels.detach().cpu()))

        return output_dict
    
    def compute_epoch_statistics(self, output_dict):
        """
        Compute epoch statistics after an epoch of training using an output_dict.
        """
        epoch_statistics = {
                            'auc' : roc_auc_score(output_dict['labels'].cpu().numpy(), 
                                                  output_dict['pred_probs'][:, 1].cpu().numpy()),
                            'auprc' : average_precision_score(output_dict['labels'].cpu().numpy(), 
                                                              output_dict['pred_probs'][:, 1].cpu().numpy()),
                            'brier' : brier_score_loss(output_dict['labels'].cpu().numpy(), 
                                                       output_dict['pred_probs'][:, 1].cpu().numpy())
                           }

        return epoch_statistics
    
    def print_metric_dict(self, metric_dict):
        print(''.join([' {}: {:4f},'.format(k, v) for k, v in metric_dict.items()]))
    
    def finalize_output_dict(self, output_dict):
        """
        Convert an output_dict to numpy
        """
        return {key: output_dict[key].cpu().numpy() for key in output_dict.keys()}
    
    def transform_batch(self, the_batch):
        """
        Sends a batch to the device
        """
        return (arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in the_batch)
    
    @staticmethod
    def weights_init(m):
        """
        Initialize the weights with Glorot initilization
        """
        if isinstance(m, nn.Linear) or \
            isinstance(m, nn.EmbeddingBag) or \
            isinstance(m, nn.Embedding) or \
            isinstance(m, SparseLinear):
            nn.init.xavier_normal_(m.weight)
    
    def init_model(self):
        """
        Initializes the model
        """
        return LinearLayer(self.config_dict['input_dim'], self.config_dict['output_dim'])
    
    def init_optimizer(self):
        """
        Initialize an optimizer
        """
        params = [{'params' : self.model.parameters()}]
        optimizer = torch.optim.Adam(params, lr = self.config_dict['lr'])
        return optimizer

    def init_scheduler(self):
        gamma = self.config_dict.get('gamma')
        if gamma is None:
            return None
        else:
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = gamma)
    
    def init_loss(self, reduction = 'mean'):
        """
        Cross entropy
        """
        return nn.CrossEntropyLoss(reduction = 'mean')
    
    def train(self, data_dict, label_dict):
        loaders = self.init_loaders(data_dict, label_dict)
        best_performance = 1e18
        loss_dict = self.init_loss_dict()
        performance_dict = self.init_performance_dict()
        
        for epoch in range(self.config_dict['num_epochs']):
            print('Epoch {}/{}'.format(epoch, self.config_dict['num_epochs'] - 1))
            print('-' * 10)

            if self.scheduler is not None:
                    self.scheduler.step()

            for phase in ['train', 'val']:
                self.model.train(phase == 'train')
                running_loss_dict = self.init_running_loss_dict(list(loss_dict[phase].keys()))
                output_dict = self.init_output_dict()
                i = 0
                for the_data in loaders[phase]:
                    # print(i)
                    i += 1
                    batch_loss_dict = {}
                    inputs, labels = self.transform_batch(the_data)
                    
                    # zero parameter gradients
                    self.optimizer.zero_grad()
                    
                    # forward
                    outputs = self.model(inputs)
                    
                    output_dict = self.update_output_dict(output_dict, outputs, labels)
                    
                    batch_loss_dict['loss'] = self.criterion(outputs, labels)
                    if phase == 'train':
                        batch_loss_dict['loss'].backward()
                        self.optimizer.step()
                    
                    for key in batch_loss_dict.keys():
                        running_loss_dict[key] += batch_loss_dict[key].item()
                    # print(running_loss_dict)
                
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
                
    def predict(self, data_dict, label_dict, phases = ['test']):
        loaders = self.init_loaders_predict(data_dict, label_dict)
        loss_dict = self.init_loss_dict(phases = phases)
        performance_dict = self.init_performance_dict(phases = phases)
        self.model.train(False)
        with torch.no_grad():
            output_dict_dict = {}
            for phase in phases:
                i = 0
                running_loss_dict = self.init_running_loss_dict(list(loss_dict[phase].keys()))
                output_dict = self.init_output_dict()
                for the_data in loaders[phase]:
                    i += 1
                    inputs, labels = self.transform_batch(the_data)
                    outputs = self.model(inputs)
                    output_dict = self.update_output_dict(output_dict, outputs, labels)
                    loss_dict['loss'] = self.criterion(outputs, labels)
                    running_loss_dict['loss'] += loss_dict['loss'].item()
                    
                # Compute epoch losses and update loss dict
                epoch_loss_dict = {key: running_loss_dict[key] / i for key in running_loss_dict.keys()}
                loss_dict[phase] = self.update_metric_dict(loss_dict[phase], epoch_loss_dict)
                # Compute epoch performance and update performance dict
                epoch_statistics = self.compute_epoch_statistics(output_dict)
                performance_dict[phase] = self.update_metric_dict(performance_dict[phase], epoch_statistics)
                output_dict_dict[phase] = self.finalize_output_dict(output_dict)
        result_dict = {phase: {**performance_dict[phase], **loss_dict[phase]} for phase in performance_dict.keys()}
        return output_dict_dict, result_dict

    def load_weights(self, the_path):
        """
        Save the model weights to a file
        """
        self.model.load_state_dict(torch.load(the_path))

    def save_weights(self, the_path):
        """
        Load model weights from a file
        """
        torch.save(self.model.state_dict(), the_path)

    @staticmethod
    def process_result_dict(the_dict, names = ['metric', 'phase', 'epoch', 'performance']):
        """
        Processes the result_dict returned from train and predict to a dataframe
        """
        result = pd.DataFrame(the_dict). \
                    reset_index(). \
                    melt(id_vars = 'index'). \
                    set_index(['index', 'variable']).value. \
                    apply(pd.Series). \
                    stack(). \
                    reset_index()
        result.columns = names
        return result

class SparseLogisticRegression(TorchModel):
    
    def init_datasets(self, data_dict, label_dict):
        """
        Creates data loaders from inputs
        """
        splits = data_dict.keys()
        dataset_dict = {key: ArrayDataset(data_dict[key], torch.LongTensor(label_dict[key]))
                                for key in splits
                        }
        return dataset_dict
    
    def init_model(self):
        layer = SparseLinear(self.config_dict['input_dim'], self.config_dict['output_dim'])
        model = SequentialLayers([layer])
        return model

class SparseLogisticRegressionEmbed(TorchModel):
    
    def init_datasets(self, data_dict, label_dict):
        """
        Creates data loaders from inputs
        """
        splits = data_dict.keys()
        dataset_dict = {key: ArrayDataset(data_dict[key], 
                                          torch.LongTensor(label_dict[key]),
                                          convert_sparse = False
                                         )
                                for key in splits
                        }
        return dataset_dict
    
    def init_model(self):
        layer = EmbeddingBagLinear(self.config_dict['input_dim'], self.config_dict['output_dim'])
        model = SequentialLayers([layer])
        return model

class FeedforwardNetModel(TorchModel):
    """
    The primary class for a feedforward net model
    """
    def init_datasets(self, data_dict, label_dict):
        """
        Creates data loaders from inputs
        """
        splits = data_dict.keys()
        dataset_dict = {key: ArrayDataset(data_dict[key], 
                                          torch.LongTensor(label_dict[key]),
                                          convert_sparse = False
                                         )
                                for key in splits
                        }
        return dataset_dict

    def init_model(self):
        model = FixedWidthClassifier(in_features = self.config_dict['input_dim'],
            hidden_dim = self.config_dict['hidden_dim'],
            num_hidden = self.config_dict['num_hidden'],
            output_dim = self.config_dict['output_dim'],
            drop_prob = self.config_dict['drop_prob'],
            normalize = self.config_dict['normalize'],
            sparse = self.config_dict['sparse'],
            sparse_mode = self.config_dict['sparse_mode'],
            resnet = self.config_dict['resnet']
            )
        return model

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

class CFVAEModel(TorchModel):
    """
    A VAE for counterfactual generation
    """
    def __init__(self, config_dict):
        # Initialize components of the generative model
        self.encoder = self.init_encoder(config_dict)
        self.decoder = self.init_decoder(config_dict)
        self.classifier = self.init_classifier(config_dict)
        self.criterion_classification = self.init_loss_classification()
        super().__init__(config_dict)

        ## Create a final_classifier to be trained after the generative model
        self.final_classifier = self.init_final_classifier(config_dict)
        self.final_classifier_optimizer = self.init_final_classifier_optim()
        self.final_classifier.apply(self.weights_init)
        self.final_classifier.to(self.device)
        
    def init_encoder(self, config_dict):
        """
        Encoder that converts data to latent representation - default to linear VAE encoder
        """
        return ConditionalLinearVAEEncoder(input_dim = config_dict['input_dim'] + config_dict['condition_embed_dim'], 
                                           latent_dim = config_dict['latent_dim'] + config_dict['condition_embed_dim'],
                                           num_conditions = config_dict['num_groups'],
                                           condition_embed_dim = config_dict['condition_embed_dim']
                                          )

    def init_decoder(self, config_dict):
        """
        Decoder that converts latent representation back to raw data
        """
        return ConditionalLinearDecoder(latent_dim = config_dict['latent_dim'],
                                        output_dim = config_dict['input_dim'],
                                        num_conditions = config_dict['num_groups'],
                                        condition_embed_dim = config_dict['condition_embed_dim']
                                        )
    
    def init_classifier(self, config_dict):
        """
        Classifier that predicts the outcome with a latent representation and conditioning information
        """
        return ConditionalLinearDecoder(latent_dim = config_dict['latent_dim'],
                                        output_dim = config_dict['output_dim'],
                                        num_conditions = config_dict['num_groups'],
                                        condition_embed_dim = config_dict['condition_embed_dim']
                                        )
    
    def init_model(self):
        """
        Initialize the CFVAE model with the components
        """
        return CFVAE(encoder = self.encoder, 
                     decoder = self.decoder, 
                     classifier = self.classifier
                    )

    def init_final_classifier(self, config_dict):
        """
        Initialize a final classifier to be trained after the generative model
        """
        return ConditionalLinearDecoder(latent_dim = config_dict['latent_dim'],
                                        output_dim = config_dict['output_dim'],
                                        num_conditions = config_dict['num_groups'],
                                        condition_embed_dim = config_dict['condition_embed_dim']
                                        )

    def init_final_classifier_optim(self):
        """
        Initialize an optimzer for the final classifier
        """
        return torch.optim.Adam(self.final_classifier.parameters(), lr = self.config_dict['lr_final_classifier'])
    
    @staticmethod
    def KL_div(mu, var):
        """
        For Gaussian with diagonal variance, compute KL divergence with standard isotropic Gaussian 
        """
        KLD = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mu.pow(2) - var, 1))
        return KLD
    
    def compute_mmd(self, x, y):
        """
        Compute an MMD
        See: https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
        """
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    @staticmethod
    def compute_kernel(x, y):
        """
        Gaussian RBF kernel for use in an MMD
        ## See https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
        """
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim) # scale-invariant gamma
        return torch.exp(-kernel_input) # (x_size, y_size)

    def compute_mmd_group(self, z, prior_z, group):
        """
        Compute the group-wise MMD, as in Louzios 2016 (http://arxiv.org/abs/1511.00830)
        """
        group_z_list = [z[group == the_group, :] for the_group in group.unique()]
        for i, z_samples in enumerate(group_z_list):
            if i == 0:
                mmd_group_loss = self.compute_mmd(z_samples, prior_z)
            else:
                mmd_group_loss = mmd_group_loss + self.compute_mmd(z_samples, prior_z)    

        return mmd_group_loss
    
    def init_loaders(self, data_dict, label_dict, group_dict):
        """
        Initialize the dataloaders for this model
        """
        dataset_dict = {key: TensorDataset(torch.FloatTensor(data_dict[key]), 
                                           torch.LongTensor(label_dict[key]),
                                           torch.LongTensor(group_dict[key])
                                          ) 
                            for key in data_dict.keys()
                       }
        
        loaders_dict = {key: DataLoader(dataset_dict[key], 
                                        batch_size = self.config_dict['batch_size']) 
                            for key in data_dict.keys()
                       }
        
        return loaders_dict
    
    def init_loss(self):
        """
        Initialize the loss for reconstruction
        """
        return nn.MSELoss()
    
    def init_loss_classification(self):
        """
        Initialize the loss for classification
        """
        return nn.CrossEntropyLoss()
    
    def init_loss_dict(self, 
        metrics = ['loss', 'elbo', 'mmd', 'reconstruction', 'kl', 'classification', 'mmd_group'], 
        phases = ['train', 'val']):
        """
        Initialize the set of losses
        For proper logging, new metrics must be passed to this method
        """
        return self.init_metric_dict(metrics = metrics, phases = phases)
    
    def init_performance_dict(self, metrics = ['auc', 'auprc', 'brier'], phases = ['train', 'val']):
        """
        Initialize non-loss metrics that are used as a part of classification
        """
        return self.init_metric_dict(metrics = metrics, phases = phases)

    def train(self, data_dict, label_dict, group_dict):
        """
        Train the generative model
        """
        best_performance = 1e18

        loaders = self.init_loaders(data_dict, label_dict, group_dict)
        loss_dict = self.init_loss_dict()
        performance_dict = self.init_performance_dict()
        self.model.train(False)
        for epoch in range(self.config_dict['num_epochs']):
            print('Epoch {}/{}'.format(epoch, self.config_dict['num_epochs'] - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                self.final_classifier.train(phase == 'train')
                
                running_loss_dict = {key : 0.0 for key in loss_dict[phase].keys()}
                output_dict = self.init_output_dict()
                i = 0
                for the_data in loaders[phase]:
                    batch_loss_dict = {}
                    i += 1
                    inputs, labels, group = self.transform_batch(the_data)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    outputs, y_outputs, mu, var, z = self.model(inputs, group)
                    
                    # Reconstruction
                    batch_loss_dict['reconstruction'] = self.criterion(outputs, inputs)
                    
                    # KL
                    batch_loss_dict['kl'] = self.KL_div(mu, var)

                    # MMD
                    prior_samples = torch.randn_like(z, requires_grad = False)
                    batch_loss_dict['mmd'] = self.compute_mmd(z, prior_samples)
                    
                    # Classification
                    batch_loss_dict['classification'] = self.criterion_classification(y_outputs, labels)
                    output_dict = self.update_output_dict(output_dict, y_outputs, labels)

                    # Group MMD (see Louizos 2016: http://arxiv.org/abs/1511.00830)
                    batch_loss_dict['mmd_group'] = self.compute_mmd_group(z, prior_samples, group)
                    
                    # Aggregate the losses
                    batch_loss_dict['loss'] = batch_loss_dict['reconstruction'] + \
                                (self.config_dict['lambda_mmd'] * batch_loss_dict['mmd']) + \
                                (self.config_dict['lambda_kl'] * batch_loss_dict['kl']) + \
                                (self.config_dict['lambda_classification'] * batch_loss_dict['classification']) + \
                                (self.config_dict['lambda_mmd_group'] * batch_loss_dict['mmd_group'])
                    # ELBO
                    batch_loss_dict['elbo'] = batch_loss_dict['reconstruction'] + batch_loss_dict['kl']
                    
                    if phase == 'train':
                        batch_loss_dict['loss'].backward()
                        self.optimizer.step()
                    
                    for key in batch_loss_dict.keys():
                        running_loss_dict[key] += batch_loss_dict[key].item()

                # Compute Losses
                epoch_loss_dict = {key: running_loss_dict[key] / i for key in running_loss_dict.keys()}
                # Update the loss dict
                loss_dict[phase] = self.update_metric_dict(loss_dict[phase], epoch_loss_dict)
                # epoch_loss_str = ''.format(phase).join([' {}: {:4f},'.format(k, v) 
                                                        # for k, v in epoch_loss_dict.items()])
                
                # Update the performance dict
                epoch_statistics = self.compute_epoch_statistics(output_dict)
                performance_dict[phase] = self.update_metric_dict(performance_dict[phase], epoch_statistics)
                # epoch_stats_str = ''.format(phase).join([' {}: {:4f},'.format(k, v) 
                                                         # for k, v in epoch_statistics.items()])
                
                print('Phase: {}:'.format(phase))
                self.print_metric_dict(epoch_loss_dict)
                self.print_metric_dict(epoch_statistics)

                if (phase == 'val') & (epoch_loss_dict['loss'] < best_performance):
                    print('Best model updated')
                    best_performance = epoch_loss_dict['loss']
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        print('Best val performance: {:4f}'.format(best_performance))

        self.model.load_state_dict(best_model_wts)
        
        result_dict = {phase: {**performance_dict[phase], **loss_dict[phase]} for phase in performance_dict.keys()}
        return result_dict

    def sample_labels(self, outputs):
        """
        Samples a label given the logits from a classifier.
        """
        pred_probs = F.softmax(outputs, dim = 1)[:, 1]
        random_mask = torch.rand_like(pred_probs)
        return (pred_probs >= random_mask).to(dtype = torch.long, device = self.device)

    def train_final_classifier(self, data_dict, label_dict, group_dict):
        """
        Train the final classifier
        """
        best_performance = 1e18

        loaders = self.init_loaders(data_dict, label_dict, group_dict)
        loss_dict = self.init_loss_dict(metrics = ['loss', 'classification', 'classification_cf', 'clp'])
        performance_dict = self.init_performance_dict()
        performance_dict_cf = self.init_performance_dict()
        self.model.train(False)

        for epoch in range(self.config_dict['num_epochs']):
            print('Epoch {}/{}'.format(epoch, self.config_dict['num_epochs'] - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                
                running_loss_dict = {key : 0.0 for key in loss_dict[phase].keys()}
                output_dict = self.init_output_dict()
                output_dict_cf = self.init_output_dict()
                i = 0
                for the_data in loaders[phase]:
                    batch_loss_dict = {}
                    i += 1
                    inputs, labels, group = self.transform_batch(the_data)

                    # zero the parameter gradients
                    self.final_classifier_optimizer.zero_grad()

                    # forward
                    z, _, _ = self.model.encoder(inputs, group)

                    # Compute the factual loss
                    y_outputs = self.final_classifier(z, group)
                    batch_loss_dict['classification'] = self.criterion_classification(y_outputs, labels)

                    # Loop through the groups - TODO - make this more efficient
                    y_cf_list = []
                    z_list = []
                    group_list = []
                    group_cf_list = []
                    y_mask_list = []

                    for the_group in range(self.config_dict['num_groups']):
                        group_cf = the_group * torch.ones_like(group)
                        
                        # Sample some y's for group_cf
                        sampled_y = self.sample_labels(self.model.classifier(z, group_cf))
                        
                        group_mask = group != group_cf # elements that are counterfactual
                        y_mask = labels == sampled_y # elements with the same label of y

                        # Accumulate relevant elements over the loop
                        y_cf_list.append(sampled_y[group_mask])
                        z_list.append(z[group_mask])
                        group_list.append(group[group_mask])
                        group_cf_list.append(group_cf[group_mask])
                        y_mask_list.append(y_mask[group_mask])

                    y_cf = torch.cat(y_cf_list, dim = 0)
                    z_cf = torch.cat(z_list, dim = 0)
                    group_cf_factual = torch.cat(group_list, dim = 0)
                    group_cf = torch.cat(group_cf_list, dim = 0)
                    y_mask = torch.cat(y_mask_list, dim = 0)

                    y_outputs_cf_factual = self.final_classifier(z_cf, group_cf_factual) # the factual y corresponding to the cf y's
                    y_outputs_cf = self.final_classifier(z_cf, group_cf)
                    batch_loss_dict['classification_cf'] = self.criterion_classification(y_outputs_cf, y_cf)
                    batch_loss_dict['clp'] = ((y_outputs_cf_factual[y_mask, :] - y_outputs_cf[y_mask, :]) ** 2).mean()

                    output_dict = self.update_output_dict(output_dict, y_outputs, labels)
                    output_dict_cf = self.update_output_dict(output_dict_cf, y_outputs_cf, y_cf)
                    batch_loss_dict['loss'] = batch_loss_dict['classification'] + \
                                                self.config_dict['lambda_final_classifier_cf'] + batch_loss_dict['classification_cf'] + \
                                                self.config_dict['lambda_clp'] + batch_loss_dict['clp']
                    
                    if phase == 'train':
                        batch_loss_dict['loss'].backward()
                        self.final_classifier_optimizer.step()
                    
                    for key in batch_loss_dict.keys():
                        running_loss_dict[key] += batch_loss_dict[key].item()

                # Compute Losses
                epoch_loss_dict = {key: running_loss_dict[key] / i for key in running_loss_dict.keys()}
                # Update the loss dict
                loss_dict[phase] = self.update_metric_dict(loss_dict[phase], epoch_loss_dict)
                
                # Update the performance dict - on factual data
                epoch_statistics = self.compute_epoch_statistics(output_dict)
                performance_dict[phase] = self.update_metric_dict(performance_dict[phase], epoch_statistics)

                # Update the performance dict - on counterfactual data
                epoch_statistics_cf = self.compute_epoch_statistics(output_dict_cf)
                performance_dict_cf[phase] = self.update_metric_dict(performance_dict_cf[phase], epoch_statistics_cf)
                
                print('Phase: {}:'.format(phase))
                
                self.print_metric_dict(epoch_loss_dict)
                print('Factual')
                self.print_metric_dict(epoch_statistics)
                print('Counterfactual')
                self.print_metric_dict(epoch_statistics_cf)

                if (phase == 'val') & (epoch_loss_dict['loss'] < best_performance):
                    print('Best model updated')
                    best_performance = epoch_loss_dict['loss']
                    best_model_wts = copy.deepcopy(self.final_classifier.state_dict())

        print('Best val performance: {:4f}'.format(best_performance))

        self.final_classifier.load_state_dict(best_model_wts)
        
        result_dict = {phase: {**performance_dict[phase], **loss_dict[phase]} for phase in performance_dict.keys()}
        return result_dict

    def predict_final_classifier(self, data_dict, label_dict, group_dict, phases = ['test']):
        """
        Train the final classifier
        """
        best_performance = 1e18

        loaders = self.init_loaders(data_dict, label_dict, group_dict)
        loss_dict = self.init_loss_dict(metrics = ['loss', 'classification', 'classification_cf', 'clp'], phases = phases)
        performance_dict = self.init_performance_dict(phases = phases)
        performance_dict_cf = self.init_performance_dict(phases = phases)

        with torch.no_grad():
            output_dict_dict = {}
            for phase in phases:
                running_loss_dict = {key : 0.0 for key in loss_dict[phase].keys()}
                output_dict = self.init_output_dict()
                output_dict_cf = self.init_output_dict()
                i = 0
                for the_data in loaders[phase]:
                    batch_loss_dict = {}
                    i += 1
                    inputs, labels, group = self.transform_batch(the_data)

                    # zero the parameter gradients
                    # self.final_classifier_optimizer.zero_grad()

                    # forward
                    z, _, _ = self.model.encoder(inputs, group)

                    # Compute the factual loss
                    y_outputs = self.final_classifier(z, group)
                    batch_loss_dict['classification'] = self.criterion_classification(y_outputs, labels)

                    # Loop through the groups - TODO - make this more efficient
                    y_cf_list = []
                    z_list = []
                    group_list = []
                    group_cf_list = []
                    y_mask_list = []

                    for the_group in range(self.config_dict['num_groups']):
                        group_cf = the_group * torch.ones_like(group)
                        
                        # Sample some y's for group_cf
                        sampled_y = self.sample_labels(self.model.classifier(z, group_cf))
                        
                        group_mask = group != group_cf # elements that are counterfactual
                        y_mask = labels == sampled_y # elements with the same label of y

                        # Accumulate relevant elements over the loop
                        y_cf_list.append(sampled_y[group_mask])
                        z_list.append(z[group_mask])
                        group_list.append(group[group_mask])
                        group_cf_list.append(group_cf[group_mask])
                        y_mask_list.append(y_mask[group_mask])

                    y_cf = torch.cat(y_cf_list, dim = 0)
                    z_cf = torch.cat(z_list, dim = 0)
                    group_cf_factual = torch.cat(group_list, dim = 0)
                    group_cf = torch.cat(group_cf_list, dim = 0)
                    y_mask = torch.cat(y_mask_list, dim = 0)

                    y_outputs_cf_factual = self.final_classifier(z_cf, group_cf_factual) # the factual y corresponding to the cf y's
                    y_outputs_cf = self.final_classifier(z_cf, group_cf)
                    batch_loss_dict['classification_cf'] = self.criterion_classification(y_outputs_cf, y_cf)
                    batch_loss_dict['clp'] = ((y_outputs_cf_factual[y_mask, :] - y_outputs_cf[y_mask, :]) ** 2).mean()

                    output_dict = self.update_output_dict(output_dict, y_outputs, labels)
                    output_dict_cf = self.update_output_dict(output_dict_cf, y_outputs_cf, y_cf)
                    batch_loss_dict['loss'] = batch_loss_dict['classification'] + \
                                                self.config_dict['lambda_final_classifier_cf'] + batch_loss_dict['classification_cf'] + \
                                                self.config_dict['lambda_clp'] + batch_loss_dict['clp']
                    
                    for key in batch_loss_dict.keys():
                        running_loss_dict[key] += batch_loss_dict[key].item()

                # Compute Losses
                epoch_loss_dict = {key: running_loss_dict[key] / i for key in running_loss_dict.keys()}
                # Update the loss dict
                loss_dict[phase] = self.update_metric_dict(loss_dict[phase], epoch_loss_dict)
                
                # Update the performance dict - on factual data
                epoch_statistics = self.compute_epoch_statistics(output_dict)
                performance_dict[phase] = self.update_metric_dict(performance_dict[phase], epoch_statistics)

                # Update the performance dict - on counterfactual data
                epoch_statistics_cf = self.compute_epoch_statistics(output_dict_cf)
                performance_dict_cf[phase] = self.update_metric_dict(performance_dict_cf[phase], epoch_statistics_cf)
                
                ## Finalize the output_dict
                output_dict = self.finalize_output_dict(output_dict)
                output_dict_dict[phase] = output_dict
        result_dict = {phase: {**performance_dict[phase], **loss_dict[phase]} for phase in performance_dict.keys()}
        return output_dict_dict, result_dict

    def save_weights_final_classifier(self, the_path):
        torch.save(self.final_classifier.state_dict(), the_path)

    def load_weights_final_classifier(self, the_path):
        self.final_classifier.load_state_dict(torch.load(the_path))