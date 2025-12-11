import numpy as np
import torch
from dataclasses import dataclass

class Domain:
    def __init__(self, name, dist_func, dist_func_torch, wrap_func, wrap_func_torch):
        self.name = name
        self.dist_func       = dist_func
        self.dist_func_torch = dist_func_torch
        self.wrap_func       = wrap_func
        self.wrap_func_torch = wrap_func_torch

    def dist(self, ps0, ps1):
        return self.dist_func(ps0, ps1)

    def dist_torch(self, ps0, ps1):
        return self.dist_func_torch(ps0, ps1)

    def wrap(self, points):
        return self.wrap_func(points)

    def wrap_torch(self, points):
        return self.wrap_func_torch(points)

def R2_dist(ps0, ps1):
    return ps1 - ps0

def R2_wrap(points):
    return points


def wrapped_square_dist(ps0, ps1):
    return np.stack([shortest_dist_wrapping_square(ps0[i], ps1[i]) for i in range(len(ps0))])

def wrapped_square_dist_torch(ps0, ps1):
    return torch.stack([shortest_dist_wrapping_square_torch(ps0[i], ps1[i]) for i in range(len(ps0))])

def wrapped_square_wrap(points):
    return points - np.floor(points)

def wrapped_square_wrap_torch(points):
    return points - torch.floor(points)


def shortest_dist_wrapping_square(x0, x1):
    # Testing: https://www.desmos.com/calculator/w0hfsd0lfe
    deltas = x1 - x0
    delta_signs = np.sign(deltas)
    flip = [1 if abs(d) > 0.5 else 0 for d in deltas]

    return deltas - delta_signs * flip

def shortest_dist_wrapping_square_torch(x0, x1):
    # Testing: https://www.desmos.com/calculator/w0hfsd0lfe
    deltas = x1 - x0
    delta_signs = torch.sign(deltas)
    flip = torch.Tensor([1 if abs(d) > 0.5 else 0 for d in deltas])

    return deltas - delta_signs * flip

R2_Domain = Domain("R2", R2_dist, R2_dist, R2_wrap, R2_wrap)

Wrapped_Square_Domain = Domain("Wrap", wrapped_square_dist, wrapped_square_dist_torch, wrapped_square_wrap, wrapped_square_wrap_torch)