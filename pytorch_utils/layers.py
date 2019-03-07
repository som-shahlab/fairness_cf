import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F



from torch.utils.data import TensorDataset, DataLoader

class LinearLayer(nn.Module):
    """
    Linear Regression model
    """
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        
    def forward(self, x):
        return self.linear(x)

class SparseLinear(nn.Module):
    """
    This is a replacement for nn.Linear where the input matrix may be a sparse tensor.
    Note that the weight attribute is defined as the transpose of the definition in nn.Linear
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        return self.sparse_linear(input, self.weight, self.bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def sparse_linear(self, input, weight, bias=None):
        output = torch.sparse.mm(input, weight)
        if bias is not None:
            output += bias
        ret = output
        return ret

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class EmbedBagLinear(nn.Module):
    __constants__ = ['bias']
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.EmbeddingBag(num_embeddings = in_features, 
                                       embedding_dim = out_features, 
                                       mode = 'sum'
                                     )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def compute_offsets(self, batch):
        return torch.LongTensor(batch.indices).to(self.weight.weight.device), \
                torch.LongTensor(batch.indptr[:-1]).to(self.weight.weight.device)

    def forward(self, x):
        return self.weight(*self.compute_offsets(x)) + self.bias

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