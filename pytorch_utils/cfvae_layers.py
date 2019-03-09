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
        z, mu, var = self.encoder(x, x_cond)
        x_reconstr = self.decoder(z, x_cond)
        y_pred = self.classifier(z, x_cond)
        return x_reconstr, y_pred, mu, var, z

class ConditionalLinearVAEEncoder(torch.nn.Module):
    """
    A conditional linear VAE encoder 
    """
    def __init__(self, input_dim, latent_dim, num_conditions, condition_embed_dim):
        super().__init__()
        self.conditional_layer = nn.Embedding(num_conditions, condition_embed_dim)
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
    
    def forward(self, x, x_cond = None):
        if x_cond is None:
            raise ValueError('x_cond must be provided')
        embedded_cond = self.conditional_layer(x_cond)
        embedded_data = torch.cat((x, embedded_cond), dim = 1)
        mu = self.latent_mean_layer(embedded_data)
        var = F.softplus(self.latent_var_layer(embedded_data)) + 1e-8
        return self.reparameterize(mu, var), mu, var
    
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