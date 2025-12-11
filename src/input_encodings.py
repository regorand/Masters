import torch
from torch import nn

class IdentityEncoding(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.encoding_size = in_features

    def forward(self, x):
        return x
    

class OneBlobEncoding(nn.Module):
    def __init__(self, in_features, K):
        super().__init__()
        self.in_features = in_features
        self.K = K
        self.encoding_size = in_features * K

        self.sigma = 1.0 / self.K
        self.register_buffer('kernel', torch.arange(1 / (2 * self.K), 1, self.sigma))

    def forward(self, x):        
        sqrt_2_pi = 2.50662827463
        values = 1 / (sqrt_2_pi * self.sigma) * torch.exp(-(x.unsqueeze(-1) - self.kernel)**2 / (2 * self.sigma**2))

        return values.flatten(start_dim=-2)
    

"""
Convience functions to build configs for tinycudann encodings
"""
def build_one_blob_encoding_conf(input_size, bins):
    return {
        "otype": "OneBlob",
        "n_dims_to_encode": input_size, 
        "n_bins": bins
    }, input_size * bins


def build_spherical_harmonics_encoding(input_size, degree):
    return {
	    "otype": "SphericalHarmonics",
        "n_dims_to_encode": input_size, 
	    "degree": degree
    }, degree * degree

def build_identity_encoding_conf(input_size):
    return {
        "otype": "Identity",
        "n_dims_to_encode": input_size, 
    }, input_size

def build_hashgrid_encoding_conf(input_size, levels, features_per_level, log2_hashmap_size = 19, base_resolution = 16, per_level_scale = 2.0):
    return {
        "n_dims_to_encode": input_size,
        "otype": "Grid",          
        "type": "Hash",           
        "n_levels": levels,  
        "n_features_per_level": features_per_level, 
        "log2_hashmap_size": log2_hashmap_size,  
        "base_resolution": base_resolution,    
        "per_level_scale": per_level_scale,   
        "interpolation": "Linear" 
    }, levels * features_per_level
