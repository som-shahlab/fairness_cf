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

class TorchModel:
    """
    This is the upper level class that provides training and logging code for a Pytorch model.
    To initialize the model, provide a config_dict with relevant parameters.
    The default model is logistic regression. Subclass and override init_model() for custom usage.

    The user is intended to interact with this class primarily through the train method.
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

    def init_datasets(self, data_dict, label_dict, **kwargs):
        """
        Method that converts data and labels to instances of class torch.utils.data.Dataset
            Args:
                data_dict: This is a dictionary that minimally contains the keys ['train', 'val']. 
                    Each element of the dictionary is the data to be converted to Dataset.
                label_dict: This is a dictionary that minimally contains the keys ['train', 'val'].
                    Each element of the dictionary are the labels to be converted to Dataset.

            Returns:
                a dictionary with the same keys as data_dict and label_dict. 
                    Each element of the dictionary is a Dataset that may be processed by torch.utils.data.DataLoader
        """
        return {key: TensorDataset(torch.FloatTensor(data_dict[key]), 
                                           torch.LongTensor(label_dict[key])) 
                            for key in data_dict.keys()
                }

    def init_loaders(self, *args, **kwargs):
        """
        Method that converts data and labels to instances of class torch.utils.data.DataLoader
            Args:
                data_dict: This is a dictionary that minimally contains the keys ['train', 'val']. 
                    Each element of the dictionary is the data to be converted to Dataset by init_datasets.
                label_dict: This is a dictionary that minimally contains the keys ['train', 'val'].
                    Each element of the dictionary are the labels to be converted to Dataset by init_loaders.

            Returns:
                a dictionary with the same keys as data_dict and label_dict. 
                    Each element of the dictionary is an instance of torch.utils.data.DataLoader
                        that yields paired elements of data and labels
        """

        # Convert the data to Dataset
        dataset_dict = self.init_datasets(*args, **kwargs)

        # If the Dataset implements collate_fn, that is used. Otherwise, default_collate is used
        if hasattr(dataset_dict['train'], 'collate_fn') and callable(getattr(dataset_dict['train'], 'collate_fn')):
            collate_fn = dataset_dict['train'].collate_fn
        else:
            collate_fn = default_collate

        # If 'iters_per_epoch' is defined, then a fixed number of random sample batches from the training set
            # are drawn per epoch.
        # Otherwise, an epoch is defined by a full run through all of the data in the dataloader.
        # 
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
        Initialize the loss_loss
        """
        return self.init_metric_dict(metrics = metrics, phases = phases)
    
    def init_running_loss_dict(self, metrics):
        """
        Initialize the running_loss_dict
        """
        return {key : 0.0 for key in metrics}
    
    def init_performance_dict(self, metrics = ['auc', 'auprc', 'brier'], phases = ['train', 'val']):
        """
        Initialize the performance_dict
        """
        return self.init_metric_dict(metrics = metrics, phases = phases)
    
    def init_output_dict(self):
        """
        Initialize the output dict
        """
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
        """
        Print method for a dictionary containing logged metrics
        """
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
        """
        A learning rate scheduler
        """
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
        """
        Method that trains the model.
            Args:
                data_dict: This is a dictionary that minimally contains the keys ['train', 'val']. 
                    Each element of the dictionary is the data that is fed to the model.
                label_dict: This is a dictionary that minimally contains the keys ['train', 'val'].
                    Each element of the dictionary are the labels corresponding to the data fed to the model.
                The type of the data and labels expected may be modified by the implementation of init_loaders.

            Returns:
                result_dict: A dictionary with metrics recorded every epoch

        """
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
                    i += 1
                    batch_loss_dict = {}
                    inputs, labels = self.transform_batch(the_data)
                    print(inputs)
                    
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
        """
        For a trained model, produce predictions on a test set and log performance.
            Args:
                data_dict: This is a dictionary that minimally contains the keys in the phases argument.
                    Type of data_dict should match that used in train.
                label_dict: This is a dictionary that minimally contains the keys in the phases argument.
                    Type of label_dict should match that used in train.

            Returns:
                output_dict_dict: A dictionary with predictions for the test samples
                result_dict: A dictionary with metrics recorded every epoch

        """
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

class FeedforwardNetModel(TorchModel):
    """
    The primary class for a feedforward network with a fixed number of hidden layers of equal size.
    Has options for sparse inputs, residual connections, dropout, and layer normalization.
    """
    def init_datasets(self, data_dict, label_dict, **kwargs):
        """
        Creates data loaders from inputs
        """
        convert_sparse = True if kwargs['sparse_mode'] == 'binary' else False
        splits = data_dict.keys()
        dataset_dict = {key: ArrayDataset(data_dict[key], 
                                          torch.LongTensor(label_dict[key]),
                                          convert_sparse = convert_sparse
                                         )
                                for key in splits
                        }
        return dataset_dict

    def init_model(self):
        model = FixedWidthNetwork(in_features = self.config_dict['input_dim'],
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

class SparseLogisticRegression(TorchModel):
    """
    A model that perform sparse logistic regression
    """
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
    """
    A model that performs sparse logistic regression with an EmbeddingBag encoder
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
        layer = EmbeddingBagLinear(self.config_dict['input_dim'], self.config_dict['output_dim'])
        model = SequentialLayers([layer])
        return model

