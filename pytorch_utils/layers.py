import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Subclasses of torch.nn.Module
"""

class FeedforwardNet(torch.nn.Module):
    """
    Feedforward network of arbitrary size
    """
    def __init__(self, in_features, hidden_dim_list = [], output_dim = 2, 
        drop_prob = 0.0, normalize = False, activation = F.leaky_relu, 
        sparse = False, sparse_mode = 'binary', resnet = False):
        super().__init__()

        num_hidden = len(hidden_dim_list)
        ## If no hidden layers - go right from input to output (equivalent to logistic regression)
        if num_hidden == 0:
            output_layer = LinearLayerWrapper(in_features, output_dim, sparse = sparse, sparse_mode = sparse_mode)
            self.layers = nn.ModuleList([output_layer])

        ## If 1 or more hidden layer, create input and output layer separately
        elif num_hidden >= 1:
            input_layer = HiddenLinearLayer(in_features = in_features,
                                                out_features = hidden_dim_list[0],
                                                drop_prob = drop_prob,
                                                normalize = normalize,
                                                activation = activation,
                                                sparse = sparse,
                                                sparse_mode = sparse_mode
                                                )
            self.layers = nn.ModuleList([input_layer])
            if resnet:
                self.layers.extend([ResidualBlock(hidden_dim = hidden_dim_list[0], 
                                                drop_prob = drop_prob,
                                                normalize = normalize,
                                                activation = activation)])

            output_layer = nn.Linear(hidden_dim_list[-1], output_dim)

            ## If more than one hidden layer, create intermediate hidden layers
            if num_hidden > 1:
                ## Standard feedforward network
                if not resnet:
                    self.layers.extend([HiddenLinearLayer(in_features = hidden_dim_list[i], 
                                                            out_features = hidden_dim_list[i+1], 
                                                            drop_prob = drop_prob,
                                                            normalize = normalize,
                                                            activation = activation,
                                                            sparse = False
                                                            ) for i in range(num_hidden - 1)])
                else: # Resnet-like architecture
                    for i in range(num_hidden - 1):
                        if hidden_dim_list[i] is not hidden_dim_list[i+1]:
                            self.layers.extend([HiddenLinearLayer(in_features = hidden_dim_list[i], 
                                                                out_features = hidden_dim_list[i+1], 
                                                                drop_prob = drop_prob,
                                                                normalize = normalize,
                                                                activation = activation,
                                                                sparse = False
                                                                )])
                        self.layers.extend([ResidualBlock(hidden_dim = hidden_dim_list[i + 1], 
                                                          drop_prob = drop_prob,
                                                          normalize = normalize,
                                                          activation = activation
                                                          )])
            self.layers.extend([output_layer])

    def forward(self, x):
        y_pred = nn.Sequential(*self.layers).forward(x)
        return y_pred

# class FixedWidthNetwork(FeedforwardNet):
#     """
#     Feedforward network with a fixed number of hidden layers of equal size.
#     """
#     def __init__(self, in_features, hidden_dim, num_hidden, output_dim = 2, 
#         drop_prob = 0.0, normalize = False, activation = F.leaky_relu, sparse = False, sparse_mode = 'binary', resnet = False):

#         # Send to FeedforwardNet
#         super().__init__(in_features = in_features, hidden_dim_list = num_hidden * [hidden_dim], 
#             output_dim = output_dim, drop_prob = drop_prob, normalize = normalize, 
#             activation = activation, sparse = sparse, sparse_mode = sparse_mode, resnet = resnet)

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