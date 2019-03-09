import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


class SequentialLayers(nn.Module):
    """
    Wraps an arbitrary list of layers with nn.Sequential.
    """
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return nn.Sequential(*self.layers).forward(x)

class LinearLayer(nn.Module):
    """
    Linear Regression model
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
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
        nn.init.xavier_normal_(self.weight)
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

class EmbeddingBagLinear(nn.Module):
    """
    This is a more efficient replacement for SparseLinear.
    Only valid if input data are binary.
    """
    __constants__ = ['bias']
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.embed = nn.EmbeddingBag(num_embeddings = in_features, 
                                       embedding_dim = out_features, 
                                       mode = 'sum'
                                     )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embed.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.embed.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def compute_offsets(self, batch):
        return torch.LongTensor(batch.indices).to(self.embed.weight.device), \
                torch.LongTensor(batch.indptr[:-1]).to(self.embed.weight.device)

    def forward(self, x):
        if self.bias is not None:
            return self.embed(*self.compute_offsets(x)) + self.bias
        else:
            return self.embed(*self.compute_offsets(x))

class HiddenLinearLayer(torch.nn.Module):
    """
    A neural network layer
    """
    def __init__(self, in_features, out_features, drop_prob = 0.0, normalize = False, activation = F.leaky_relu, sparse = False, sparse_mode = 'binary'):
        super().__init__()

        self.linear = LinearLayerWrapper(in_features, out_features, sparse = sparse, sparse_mode = sparse_mode)
        self.dropout = nn.Dropout(p = drop_prob)
        self.activation = activation
        self.normalize = normalize
        if self.normalize:
            self.normalize_layer = nn.LayerNorm(normalized_shape = out_features)
        
    def forward(self, x):
        if self.normalize:
            result = self.dropout(self.activation(self.normalize_layer(self.linear(x))))
        else:
            result = self.dropout(self.activation(self.linear(x)))
        return result

class LinearLayerWrapper(torch.nn.Module):
    """
    Wrapper around various linear layers to call appropriate sparse layer
    """
    def __init__(self, in_features, out_features, sparse = False, sparse_mode = 'binary'):
        super().__init__()
        if sparse and (sparse_mode == 'binary'):
            self.linear = EmbeddingBagLinear(in_features, out_features)
        elif sparse and (sparse_mode == 'count'):
            self.linear = SparseLinear(in_features, out_features)
        else:
            self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear.forward(x)

class ResidualBlock(torch.nn.Module):
    """
    A residual block for fully connected networks
    """
    def __init__(self, hidden_dim, drop_prob = 0.0, normalize = False, activation = F.leaky_relu):
        super().__init__()

        self.layer1 = HiddenLinearLayer(in_features = hidden_dim, out_features = hidden_dim, 
            drop_prob = drop_prob, normalize = normalize, activation = activation)
        self.layer2 = HiddenLinearLayer(in_features = hidden_dim, out_features = hidden_dim, 
            drop_prob = drop_prob, normalize = normalize, activation = self.identity)

        self.activation = activation

    def forward(self, x):
        result = self.activation(self.layer2(self.layer1(x)) + x)
        return result

    def identity(self, x):
        return x

class FixedWidthClassifier(torch.nn.Module):
    """
    Feedforward network with a fixed number of hidden layers. Handles sparse input
    """
    def __init__(self, in_features, hidden_dim, num_hidden, output_dim = 2, 
        drop_prob = 0.0, normalize = False, activation = F.leaky_relu, sparse = False, sparse_mode = 'binary', resnet = False):
        super().__init__()

        ## If no hidden layers - go right from input to output
        if num_hidden == 0:
            self.output_layer = LinearLayerWrapper(in_features, output_dim, sparse = sparse, sparse_mode = sparse_mode)
            self.layers = nn.ModuleList([self.output_layer])

        ## If 1 or more hidden layer, create input and output layer separately
        elif num_hidden >= 1:
            self.input_layer = HiddenLinearLayer(in_features = in_features,
                                                out_features = hidden_dim,
                                                drop_prob = drop_prob,
                                                normalize = normalize,
                                                activation = activation,
                                                sparse = sparse,
                                                sparse_mode = sparse_mode
                                                )
            self.layers = nn.ModuleList([self.input_layer])
            self.output_layer = nn.Linear(hidden_dim, output_dim)

            ## If more than one hidden layer, create intermediate hidden layers
            if num_hidden > 1:
                if resnet:
                    self.layers.extend([ResidualBlock(hidden_dim = hidden_dim, 
                                                      drop_prob = drop_prob,
                                                      normalize = normalize,
                                                      activation = activation
                                                      ) for i in range(num_hidden - 1)])
                else:
                    self.layers.extend([HiddenLinearLayer(in_features = hidden_dim, 
                                                            out_features = hidden_dim, 
                                                            drop_prob = drop_prob,
                                                            normalize = normalize,
                                                            activation = activation,
                                                            sparse = False
                                                            ) for i in range(num_hidden - 1)])
            self.layers.extend([self.output_layer])

    def forward(self, x):
        y_pred = nn.Sequential(*self.layers).forward(x)
        return y_pred

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