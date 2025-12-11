import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from math import log, sin, cos, sqrt, pi


""" <Image Utils> """
def gauss_blur(input, sigma):
    return gaussian_filter(input, sigma=sigma)

def filter_and_norm(input, sigma):
    filtered = gauss_blur(input.astype(np.float64), sigma)
    return np.divide(filtered, np.max(filtered))

def points_to_image(points, dims):
    if type(points) == torch.Tensor:
        points = points.detach().numpy()

    dims = np.array([dims[0], dims[1]])
    rev_dims = np.array([dims[1], dims[0]])
    dist_test = np.zeros(dims)
    for point in points:

        coords = np.clip((point * rev_dims).astype(np.int32), [0], rev_dims - 1)
    
        dist_test[coords[1]][coords[0]] += 1

    return dist_test
""" </Image Utils> """


def sample_uniform_points(count: int):
    return np.random.rand(count, 2)

def sample_uniform_points_torch(count):
    return torch.rand(count, 2)

def box_mueller_transform(unif_points):
    out_points = []
    for p in unif_points:
        Z0 = sqrt(-2 * log(p[0])) * cos(2 * pi * p[1])
        Z1 = sqrt(-2 * log(p[0])) * sin(2 * pi * p[1])
        out_points.append((Z0, Z1))
    return out_points


def point_to_unit_sphere(point: np.ndarray):
    phi = point[0] * 2 * pi             # azimuth angle
    psi = - (point[1] * 2 - 1) * pi  / 2  # polar angle
    return torch.Tensor([cos(phi) * cos(psi), sin(phi) * cos(psi), sin(psi)])

def points_to_unit_sphere(points: torch.Tensor):
    # if type(points) == np.ndarray:
    #     points = torch.from_numpy(points)
        
    phi = points[:, 0] * 2 * torch.pi
    psi = (points[:, 1] * 2 - 1) * torch.pi / 2

    x = torch.cos(phi) * torch.cos(psi)
    y = torch.sin(phi) * torch.cos(psi)
    z = torch.sin(psi)

    return torch.stack((x, y, z), dim = 1)
    # l = [point_to_unit_sphere(p) for p in points]
    # return torch.stack(l)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
