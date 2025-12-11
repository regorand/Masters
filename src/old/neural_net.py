from torch import nn
import torch
from math import log


# Implementation originally based on the example: https://www.youtube.com/watch?v=7cMzfkWFWhI

""" 
Embedding of the time factor into logarithmically spaced frequencies fed into sin/cos function

This will be fed through an input projection and added to the result of the input values in projection
"""

def log_frequency_time_embedding(t, channels_t, max_freq_positions = 10000):
    t = t * max_freq_positions
    half_dim = channels_t // 2
    emb = log(max_freq_positions) / (half_dim - 1)
    emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    if channels_t % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1), mode='constant')
    return emb


class Layer(nn.Module):
    def __init__(self, channels_in, channels_out, activation = nn.ReLU()):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.lin_forward = nn.Linear(self.channels_in, self.channels_out)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.lin_forward(x))


class MLP(nn.Module):
    """
    Represents the actual neural network part of a model. 
    Does not hold the extra information required to correctly operate it (e.g. input distribtion, domain, etc.)

    
    Attributes
    ---------
    layers : [int]
        Array of ints specifying hidden layers. N entries will result in N-1 layers.
    activation_function : Callable
        Activation function to be used in the network, as of yet not configurable by layer
    """

    def __init__(self, 
                 channels_in, 
                 channels_out, 
                 channels_t, 
                 layers, 
                 activation_function = nn.ReLU(), 
                 time_embedding = log_frequency_time_embedding):
        super().__init__()
        if len(layers) < 1:
            raise Exception("Need at least 2 entries in layer description to create MLP")
        
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.channels_t = channels_t
        
        self.net = nn.Sequential(*[
            Layer(layer_in, layer_out, activation_function) for layer_in, layer_out in zip(layers, layers[1:])
        ])

        self.in_proj = nn.Linear(channels_in, layers[0])
        self.t_proj = nn.Linear(channels_t, layers[0])
        self.out_proj = nn.Linear(layers[-1], channels_out)

        self.time_embedding = time_embedding

    def forward(self, x, t):
        x = self.in_proj(x)
        t = self.time_embedding(t, channels_t = self.channels_t)
        t = self.t_proj (t)
        x = x + t
        x = self.net(x)
        return self.out_proj(x)
    
    def input_shape(self):
        return [[self.channels_in], 1]