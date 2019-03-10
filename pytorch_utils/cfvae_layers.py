import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class CFVAE(torch.nn.Module):
    """
    Module that defines a wrapper around the components of a counterfactual VAE
    """
    def __init__(self, encoder, decoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        
    def forward(self, x, x_cond):
        z, mu, var = self.encoder(x)
        x_reconstr = self.decoder(z, x_cond)
        y_pred = self.classifier(z, x_cond)
        return x_reconstr, y_pred, mu, var, z

class ReparameterizationLayer(torch.nn.Module):
    """
    Reparameterizes the input with Gaussian noise
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_mean_layer = nn.Linear(input_dim,
                                            latent_dim
                                          )
        self.latent_var_layer = nn.Linear(input_dim,
                                            latent_dim
                                         )
    def reparameterize(self, mu, var):
        """
        More stable reparameterization following softplus activation
        """
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu = self.latent_mean_layer(x)
        var = F.softplus(self.latent_var_layer(x)) + 1e-8
        return self.reparameterize(mu, var), mu, var

class VAEEncoder(torch.nn.Module):
    def __init__(self, encoder, reparameterization_layer):
        super().__init__()
        self.encoder = encoder
        self.reparameterization_layer = reparameterization_layer

    def forward(self, x):
        return self.reparameterization_layer(self.encoder(x))

class ConditionalDecoder(torch.nn.Module):
    """
    Wraps a decoder with a new decoder that uses conditioning information.
    Decoder definition is expected to account for dimensionality change
    """
    def __init__(self, decoder, num_conditions, condition_embed_dim):
        super().__init__()
        self.decoder = decoder
        self.conditional_layer = nn.Embedding(num_conditions, condition_embed_dim)

    def forward(self, z, x_cond):
        embedded_data = torch.cat((z, self.conditional_layer(x_cond)), dim = 1)
        return self.decoder(embedded_data)

# class FixedWidthVAEEncoder(torch.nn.Module):
#     def __init__():

#         FixedWidthNetwork(in_features = self.config_dict['input_dim'],
#             hidden_dim = self.config_dict['hidden_dim'],
#             num_hidden = self.config_dict['num_hidden'],
#             output_dim = self.config_dict['latent_dim'],
#             drop_prob = self.config_dict['drop_prob'],
#             normalize = self.config_dict['normalize'],
#             sparse = self.config_dict['sparse'],
#             sparse_mode = self.config_dict['sparse_mode'],
#             resnet = self.config_dict['resnet']
#             )

# class ConditionalLinearVAEEncoder(torch.nn.Module):
#     """
#     A conditional linear VAE encoder 
#     """
#     def __init__(self, input_dim, latent_dim, num_conditions, condition_embed_dim):
#         super().__init__()
#         self.conditional_layer = nn.Embedding(num_conditions, condition_embed_dim)
#         self.latent_mean_layer = nn.Linear(input_dim,
#                                             latent_dim
#                                           )
#         self.latent_var_layer = nn.Linear(input_dim,
#                                             latent_dim
#                                          )
#     def reparameterize(self, mu, var):
#         """
#         More stable reparameterization following softplus activation
#         """
#         std = torch.sqrt(var)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu)
    
#     def forward(self, x, x_cond = None):
#         if x_cond is None:
#             raise ValueError('x_cond must be provided')
#         embedded_cond = self.conditional_layer(x_cond)
#         embedded_data = torch.cat((x, embedded_cond), dim = 1)
#         mu = self.latent_mean_layer(embedded_data)
#         var = F.softplus(self.latent_var_layer(embedded_data)) + 1e-8
#         return self.reparameterize(mu, var), mu, var
    
class ConditionalLinearDecoder(torch.nn.Module):
    """
    A conditional linear decoder
    """
    def __init__(self, latent_dim, output_dim, num_conditions, condition_embed_dim):
        super().__init__()
        self.conditional_layer = nn.Embedding(num_conditions, condition_embed_dim)
        self.decoder_layer = nn.Linear(latent_dim + num_conditions, output_dim)
        
    def forward(self, z, x_cond):
        embedded_data = torch.cat((z, self.conditional_layer(x_cond)), dim = 1)
        return self.decoder_layer(embedded_data)