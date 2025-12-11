import torch
import mitsuba as mi
import numpy as np
from coordinate_system import CoordinateSystem


##################################################################
##################################################################
# Coordinate Transforms
##################################################################
##################################################################


def cartesian_to_spherical(vectors: torch.Tensor):
    y = vectors[:, 1].clamp(-1.0, 1.0)
    theta = y.acos()
    phi = torch.atan2(vectors[:, 2], vectors[:, 0])
    return torch.stack((phi, theta), dim = -1)

def spherical_to_cartesian(angles: torch.Tensor):
    phi = angles[:, 0]
    theta = angles[:, 1]

    return torch.stack((
        theta.sin() * phi.cos(),
        theta.cos(),
        theta.sin() * phi.sin()
    ), dim = -1)

def cartesian_to_cylindrical(vectors: torch.Tensor):
    y = -vectors[:, 1]
    phi = torch.atan2(vectors[:, 2], vectors[:, 0])
    return torch.stack((phi, y), dim = -1)

def cylindrical_to_cartesian(cylindrical: torch.Tensor):
    phi = cylindrical[:, 0]
    y = -cylindrical[:, 1]

    # Can compute rho implicitely because we only transform from and to directions in 3d, which are normalized
    rho = torch.sqrt(1 - y.square())

    return torch.stack((
        rho * phi.cos(),
        y,
        rho * phi.sin()
    ), dim = -1)

def spherical_to_cylindrical(angles: torch.Tensor):
    phi = angles[:, 0]
    theta = angles[:, 1]

    y = theta.cos()

    return torch.stack((phi, theta), dim = -1)

def cylindrical_to_spherical(cylindrical: torch.Tensor):
    phi = cylindrical[:, 0]
    y = cylindrical[:, 1]

    theta = y.acos()

    return torch.stack((phi, theta), dim = -1)


def cartesian_to_system(vectors: torch.Tensor, coordsSystem: CoordinateSystem):   
    if coordsSystem == CoordinateSystem.Cylindrical:
        return cartesian_to_cylindrical(vectors)
    elif coordsSystem == CoordinateSystem.Spherical:
        return cartesian_to_spherical(vectors)

def system_to_cartesian(coordinates: torch.Tensor, coordsSystem: CoordinateSystem):
    if coordsSystem == CoordinateSystem.Cylindrical:
        return cylindrical_to_cartesian(coordinates)
    elif coordsSystem == CoordinateSystem.Spherical:
        return spherical_to_cartesian(coordinates)


##################################################################
##################################################################
# Coordinate Transform jacobians
##################################################################
##################################################################

def spherical_to_cartesian_jacobian(angles: torch.Tensor):
    return angles[:, 1].sin()

def cylindrical_to_cartesian_jacobian(values: torch.Tensor):
    return torch.sqrt(1 - values[:, 1]) # This is implicit rho because usecase limits 3d vectors to directions

def system_to_cartesian_jacobian(samples: torch.Tensor, coordsSystem: CoordinateSystem):
    if coordsSystem == CoordinateSystem.Cylindrical:
        return cylindrical_to_cartesian_jacobian(samples)
    elif coordsSystem == CoordinateSystem.Spherical:
        return spherical_to_cartesian_jacobian(samples)


##################################################################
##################################################################
# Coordinate Transform domains
##################################################################
##################################################################

def get_domain_from_coords(coordinate_system: CoordinateSystem, device: torch.device):
    if coordinate_system == CoordinateSystem.Cylindrical:
        return torch.as_tensor([[-torch.pi, torch.pi], [-1, 1]], device=device)
    elif coordinate_system == CoordinateSystem.Spherical:
        return torch.as_tensor([[-torch.pi, torch.pi], [0, torch.pi]], device=device)

def get_plt_ticks_from_coords(coordinate_system: CoordinateSystem):
    if coordinate_system == CoordinateSystem.Cylindrical:
        return  [
                    [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$-\pi$"], 
                    [r"$-1$", r"$-0.5$", r"$0$", r"$0.5$", r"$1$"]
                ]
    elif coordinate_system == CoordinateSystem.Spherical:
        return  [
                    [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$-\pi$"],
                    [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{2}$", r"$\pi$"]
                ]


##################################################################
##################################################################
# drjit -> torch utilities
##################################################################
##################################################################


def res_to_numpy_3(res, width, height):
    tensor: mi.TensorXf = mi.TensorXf(res)
    # return np.ascontiguousarray(np.array(tensor, dtype=np.float32)).reshape(3, height, width).transpose((1, 2, 0))
    return  tensor.numpy().reshape(3, height, width).transpose((1, 2, 0))

def res_to_numpy_2(res, width, height):
    tensor: mi.TensorXf = mi.TensorXf(res)
    # return np.array(tensor, dtype=np.float32).reshape(2, height, width).transpose((1, 2, 0))
    return tensor.numpy().reshape(2, height, width).transpose((1, 2, 0))

def res_to_numpy_1(res, width, height):
    tensor: mi.TensorXf = mi.TensorXf(res)
    # return np.array(tensor, dtype=np.float32).reshape(1, height, width).transpose((1, 2, 0))
    return tensor.numpy().reshape(height, width).transpose((0, 1))


##################################################################
##################################################################
# Image utilities
##################################################################
##################################################################


def tonemap(value):
    return value / (value + 1)

def gamma_correct(value):
    return value ** (1.0 / 2.2)