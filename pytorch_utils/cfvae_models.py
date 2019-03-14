import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing
import scipy

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from .layers import *
from .datasets import *
from .models import *
from .cfvae_layers import *

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
        Encoder that converts data to latent representation. Should return an instance of class VAEEncoder
        """
        hidden_dim_list = [config_dict['latent_dim'] * \
            (2**(i + 1)) for i in reversed(range(config_dict['num_hidden']))]
        encoder = FeedforwardNet(in_features = config_dict['input_dim'] + config_dict['num_groups'],
            hidden_dim_list = hidden_dim_list,
            output_dim = config_dict['latent_dim']*2, ## for mu, var in latent
            drop_prob = config_dict['drop_prob'],
            normalize = config_dict['normalize'],
            sparse = config_dict['sparse'],
            sparse_mode = config_dict['sparse_mode'],
            resnet = config_dict['resnet']
            )
        reparameterization_layer = ReparameterizationLayer()
        return VAEEncoder(encoder = encoder, reparameterization_layer = reparameterization_layer)

    def init_decoder(self, config_dict):
        """
        Decoder that converts latent representation back to raw data
        """
        hidden_dim_list = [(config_dict['latent_dim'] + config_dict['group_embed_dim']) * \
            (2**(i + 1)) for i in range(config_dict['num_hidden'])]
        decoder = FeedforwardNet(in_features = config_dict['latent_dim'] + config_dict['group_embed_dim'],
            hidden_dim_list = hidden_dim_list,
            output_dim = config_dict['input_dim'],
            drop_prob = config_dict['drop_prob'],
            normalize = config_dict['normalize'],
            sparse = False,
            resnet = config_dict['resnet']
            )
        return ConditionalDecoder(decoder, 
            num_conditions = config_dict['num_groups'], 
            condition_embed_dim = config_dict['group_embed_dim'])
    
    def init_classifier(self, config_dict):
        """
        Classifier that predicts the outcome with a latent representation and conditioning information
        """
        decoder = FeedforwardNet(in_features = config_dict['latent_dim'] + config_dict['group_embed_dim'],
            hidden_dim_list = config_dict['num_hidden_classifier'] * [config_dict['hidden_dim_classifier']],
            output_dim = config_dict['output_dim'],
            drop_prob = config_dict['drop_prob_classifier'],
            normalize = config_dict['normalize_classifier'],
            sparse = False,
            resnet = config_dict['resnet_classifier']
            )
        return ConditionalDecoder(decoder, 
            num_conditions = config_dict['num_groups'], 
            condition_embed_dim = config_dict['group_embed_dim'])
    
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
        decoder = FeedforwardNet(in_features = config_dict['latent_dim'] + config_dict['group_embed_dim'],
            hidden_dim_list = config_dict['num_hidden_classifier'] * [config_dict['hidden_dim_classifier']],
            output_dim = config_dict['output_dim'],
            drop_prob = config_dict['drop_prob_classifier'],
            normalize = config_dict['normalize_classifier'],
            sparse = False,
            resnet = config_dict['resnet_classifier']
            )
        return ConditionalDecoder(decoder, 
            num_conditions = config_dict['num_groups'], 
            condition_embed_dim = config_dict['group_embed_dim'])

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

    def init_datasets(self, data_dict, label_dict, group_dict):
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
        return {key: ArrayDataset(data_dict[key], 
                                  torch.LongTensor(label_dict[key]),
                                  torch.LongTensor(group_dict[key]),
                                  convert_sparse = False
                                  ) 
                            for key in data_dict.keys()
                }

    def init_loss(self):
        """
        Initialize the loss for reconstruction
        """
        return nn.BCEWithLogitsLoss()
    
    def init_loss_classification(self):
        """
        Initialize the loss for classification
        """
        return nn.CrossEntropyLoss(reduction = 'mean')
    
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

    def init_binarizer(self, group_dict):
        binarizer = sklearn.preprocessing.LabelBinarizer(sparse_output = True)
        binarizer.fit(group_dict['train'])
        return binarizer

    def train(self, data_dict, label_dict, group_dict):
        """
        Train the generative model
        """
        best_performance = 1e18

        loaders = self.init_loaders(data_dict, label_dict, group_dict)
        group_binarizer = self.init_binarizer(group_dict)
        loss_dict = self.init_loss_dict()
        performance_dict = self.init_performance_dict()
        self.final_classifier.train(False)
        for epoch in range(self.config_dict['num_epochs']):
            print('Epoch {}/{}'.format(epoch, self.config_dict['num_epochs'] - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                self.model.train(phase == 'train')
                # self.final_classifier.train(phase == 'train')
                
                running_loss_dict = {key : 0.0 for key in loss_dict[phase].keys()}
                output_dict = self.init_output_dict()
                i = 0
                for the_data in loaders[phase]:
                    batch_loss_dict = {}
                    i += 1
                    
                    inputs, labels, group = self.transform_batch(the_data)

                    # Compute the autoencoder target based on the CSR input
                    target = torch.FloatTensor(inputs.todense()).to(self.device)

                    # Combine the inputs with the group data
                    combined_inputs = scipy.sparse.hstack((inputs, 
                        group_binarizer.transform(group.cpu().numpy())), format = 'csr')

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    outputs, y_outputs, mu, var, z = self.model(combined_inputs, group)
                    
                    # Reconstruction
                    batch_loss_dict['reconstruction'] = self.criterion(outputs, target)
                    
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
                    batch_loss_dict['loss'] = (self.config_dict['lambda_reconstruction'] * batch_loss_dict['reconstruction']) + \
                                (self.config_dict['lambda_mmd'] * batch_loss_dict['mmd']) + \
                                (self.config_dict['lambda_kl'] * batch_loss_dict['kl']) + \
                                (self.config_dict['lambda_classification'] * batch_loss_dict['classification']) + \
                                (self.config_dict['lambda_mmd_group'] * batch_loss_dict['mmd_group'])
                    # ELBO
                    batch_loss_dict['elbo'] = batch_loss_dict['reconstruction'] + batch_loss_dict['kl'] + batch_loss_dict['classification']
                    
                    if phase == 'train':
                        batch_loss_dict['loss'].backward()
                        self.optimizer.step()
                    
                    for key in batch_loss_dict.keys():
                        running_loss_dict[key] += batch_loss_dict[key].item()

                # Compute Losses
                epoch_loss_dict = {key: running_loss_dict[key] / i for key in running_loss_dict.keys()}
                # Update the loss dict
                loss_dict[phase] = self.update_metric_dict(loss_dict[phase], epoch_loss_dict)
                
                # Update the performance dict
                epoch_statistics = self.compute_epoch_statistics(output_dict)
                performance_dict[phase] = self.update_metric_dict(performance_dict[phase], epoch_statistics)
                
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

    def predict(self, data_dict, label_dict, group_dict, phases = ['test']):
        """
        Train the generative model
        """
        loaders = self.init_loaders(data_dict, label_dict, group_dict)
        loss_dict = self.init_loss_dict(phases = phases)
        performance_dict = self.init_performance_dict(phases = phases)
        self.model.train(False)
        with torch.no_grad():
            output_dict_dict = {}
            for phase in phases:
                running_loss_dict = {key : 0.0 for key in loss_dict[phase].keys()}
                output_dict = self.init_output_dict()
                i = 0
                for the_data in loaders[phase]:
                    batch_loss_dict = {}
                    i += 1
                    inputs, labels, group = self.transform_batch(the_data)

                    # Compute the autoencoder target based on the CSR input
                    target = torch.FloatTensor(inputs.todense()).to(self.device)

                    # forward
                    outputs, y_outputs, mu, var, z = self.model(inputs, group)
                    output_dict = self.update_output_dict(output_dict, y_outputs, labels)

                    # Reconstruction
                    batch_loss_dict['reconstruction'] = self.criterion(outputs, target)
                    
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
                    batch_loss_dict['loss'] = (self.config_dict['lambda_reconstruction'] * batch_loss_dict['reconstruction']) + \
                                (self.config_dict['lambda_mmd'] * batch_loss_dict['mmd']) + \
                                (self.config_dict['lambda_kl'] * batch_loss_dict['kl']) + \
                                (self.config_dict['lambda_classification'] * batch_loss_dict['classification']) + \
                                (self.config_dict['lambda_mmd_group'] * batch_loss_dict['mmd_group'])
                    # ELBO
                    batch_loss_dict['elbo'] = batch_loss_dict['reconstruction'] + batch_loss_dict['kl'] + batch_loss_dict['classification']
                    
                    
                    for key in batch_loss_dict.keys():
                        running_loss_dict[key] += batch_loss_dict[key].item()

                # Compute Losses
                epoch_loss_dict = {key: running_loss_dict[key] / i for key in running_loss_dict.keys()}
                # Update the loss dict
                loss_dict[phase] = self.update_metric_dict(loss_dict[phase], epoch_loss_dict)
                
                # Update the performance dict
                epoch_statistics = self.compute_epoch_statistics(output_dict)
                performance_dict[phase] = self.update_metric_dict(performance_dict[phase], epoch_statistics)
                output_dict_dict[phase] = self.finalize_output_dict(output_dict)

        result_dict = {phase: {**performance_dict[phase], **loss_dict[phase]} for phase in performance_dict.keys()}
        return output_dict_dict, result_dict

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
                self.final_classifier.train(phase == 'train')
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
                    z, _, _ = self.model.encoder(inputs)

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