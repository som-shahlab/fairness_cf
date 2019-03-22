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
    Partitions input into two slices as mu, var and reparameterizes
    """

    def reparameterize(self, mu, var):
        """
        More stable reparameterization following softplus activation
        """
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu = x[:, : x.shape[-1] // 2]
        var = F.softplus(x[:, x.shape[-1] // 2 :]) + 1e-8
        return self.reparameterize(mu, var), mu, var


class VAEEncoder(torch.nn.Module):
    def __init__(self, encoder, reparameterization_layer):
        super().__init__()
        self.encoder = encoder
        self.reparameterization_layer = reparameterization_layer

    def forward(self, x):
        return self.reparameterization_layer(self.encoder(x))


class ConditionalEncoder(torch.nn.Module):
    """
    Wraps an encoder with a new encoder that usses conditioning information
    """


class ConditionalDecoder(torch.nn.Module):
    """
    Wraps a decoder with a new decoder that uses conditioning information.
    Encoder definition is expected to account for dimensionality change
    """

    def __init__(self, decoder, num_conditions, condition_embed_dim):
        super().__init__()
        self.decoder = decoder
        self.conditional_layer = nn.Embedding(num_conditions, condition_embed_dim)

    def forward(self, z, x_cond):
        embedded_data = torch.cat((z, self.conditional_layer(x_cond)), dim=1)
        return self.decoder(embedded_data)


class ConditionalLinearDecoder(torch.nn.Module):
    """
    A conditional linear decoder
    """

    def __init__(self, latent_dim, output_dim, num_conditions, condition_embed_dim):
        super().__init__()
        self.conditional_layer = nn.Embedding(num_conditions, condition_embed_dim)
        self.decoder_layer = nn.Linear(latent_dim + num_conditions, output_dim)

    def forward(self, z, x_cond):
        embedded_data = torch.cat((z, self.conditional_layer(x_cond)), dim=1)
        return self.decoder_layer(embedded_data)
