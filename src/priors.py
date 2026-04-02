from numpy import apply_along_axis
import torch
import math
import utils as utils
from coordinate_system import CoordinateSystem

class Prior(torch.nn.Module):
    def sample(self, num_samples, normals = None, device = 'cuda'):
        raise NotImplementedError("Not Implemented")
    
    def prob(self, x, normals = None, device = 'cuda'):
        raise NotImplementedError("Not Implemented")

    def log_prob(self, x, normals = None, device = 'cuda'):
        raise NotImplementedError("Not Implemented")
    
    def requires_normal(self):
        raise NotImplementedError("Not Implemented")
    
class UniformPrior(Prior):
    def __init__(self, domain, coordinate_system: CoordinateSystem):
        super().__init__()
        self.register_buffer("domain", torch.tensor(domain))
        self.dim = domain.shape[0]
        self.pdf = 1 / (domain[:, 1] - domain[:, 0]).prod(dim = 0)
        self.eps = 1e-8
        self.requires_nrml = False
        self.coordinate_system = coordinate_system

    def sample(self, num_samples, normals = None, device = 'cuda'):
        return torch.rand((num_samples, self.dim), device =self.domain.device) * (self.domain[:, 1] - self.domain[:, 0]) + self.domain[:, 0]
    
    def sample_with_pdf(self, num_samples, normals = None, device = 'cuda'):
        samples = self.sample(num_samples=num_samples, normals=normals)
        pdf = torch.full((num_samples,), self.pdf, device = self.domain.device)

        return samples, pdf


    def prob(self, x, normals = None, device = 'cuda'):
        return torch.full((x.shape[0], ), self.pdf, device = self.domain.device)

    def log_prob(self, x, normals = None, device = 'cuda'):
        return (self.prob(x, normals) + self.eps).log()

    def requires_normal(self):
        return self.requires_nrml

class UniformSphere(Prior):
    def __init__(self, coordinate_system: CoordinateSystem = CoordinateSystem.Spherical):
        super().__init__()
        self.dim = 3
        self.pdf = 0.07957747 # = 1 / (4 * pi) -> i.e. 1/A where A is surface area of unit sphere
        self.eps = 1e-8
        self.requires_nrml = False
        self.coordinate_system = coordinate_system

    def sample(self, num_samples, normals = None, device = 'cuda'):
        unnorm = torch.randn((num_samples, self.dim), device=device)
        return utils.cartesian_to_spherical(torch.nn.functional.normalize(unnorm, dim = 1))
    
    def sample_with_pdf(self, num_samples, normals = None, device = 'cuda'):
        samples = self.sample(num_samples=num_samples, normals=normals)
        pdf = torch.full((num_samples,), self.pdf, device = device)

        pdf = pdf * samples[:, 1].sin()

        return samples, pdf

    def prob(self, x, normals = None, device = 'cuda'):
        return torch.full((x.shape[0], ), self.pdf, device = device) * x[:, 1].sin()

    def log_prob(self, x, normals = None, device = 'cuda'):
        return (self.prob(x, normals) + self.eps).log()

    def requires_normal(self):
        return self.requires_nrml

class CosineWeighted(Prior):
    def __init__(self, coordinate_system: CoordinateSystem):
        super().__init__()
        self.eps = 1e-6
        self.requires_nrml = True
        self.coordinate_system = coordinate_system

    def sample_local(self, num_samples, device):
        theta = torch.rand((num_samples), device = device).sqrt().acos()
        phi = torch.rand((num_samples), device = device) * 2 * torch.pi

        # TODO sample in spherical, transform to cartesian, can also transform to target coords later
        return utils.spherical_to_cartesian(torch.stack((phi, theta), dim = -1))
        # return utils.to_vec3_torch(phi, theta)

    def build_ONB(self, N):
        N = torch.nn.functional.normalize(N, dim=-1)

        cond = (torch.abs(N[:, 0]) > torch.abs(N[:, 2]))

        h = torch.where(
            cond.unsqueeze(-1),
            torch.stack([-N[:, 1], N[:, 0], torch.zeros_like(N[:, 0])], dim=-1),
            torch.stack([torch.zeros_like(N[:, 0]), -N[:, 2], N[:, 1]], dim=-1)
        )

        T = torch.nn.functional.normalize(h, dim=-1)
        B = torch.cross(N, T, dim=-1)

        onb = torch.stack((T, N, B), dim=-1)
        return onb


    def apply_ONB(self, samples, onb):
        world_dirs = (onb @ samples.unsqueeze(-1)).squeeze(-1)

        world_dirs = torch.nn.functional.normalize(world_dirs, dim=-1)
        return world_dirs
    

    def sample(self, num_samples, normals, device = 'cuda'):
        samples = self.sample_local(num_samples, normals.device)
        onb = self.build_ONB(normals)
        rotated_samples = self.apply_ONB(samples, onb)

        return utils.cartesian_to_system(rotated_samples, self.coordinate_system)
    
    def sample_with_pdf(self, num_samples, normals, device = 'cuda'):
        samples = self.sample_local(num_samples, normals.device)
        onb = self.build_ONB(normals)
        rotated_samples = self.apply_ONB(samples, onb)

        # phi, theta = utils.to_angles_torch(rotated_samples)
        angles = utils.cartesian_to_system(rotated_samples, self.coordinate_system)
        dot = (rotated_samples * normals).sum(dim=-1)

        pdf = torch.clamp(dot, min = 0.0) / torch.pi
        

        pdf = pdf * angles[:, 1].sin()

        return angles, pdf
    
    def prob(self, x, normals, device = 'cuda'):
        # vecs = utils.to_vec3_torch(x[:, 0], x[:, 1])
        vecs = utils.system_to_cartesian(x, self.coordinate_system)
        dot = (vecs * normals).sum(dim=-1)
        pdf = torch.clamp(dot, min = 0.0) / torch.pi
        
        return pdf * torch.sin(x[:, 1])

    def log_prob(self, x, normals):
        return (self.prob(x, normals) + self.eps).log()

    def requires_normal(self):
        return self.requires_nrml


class DiagGaussianPrior(Prior):
    def __init__(self, means, stds, coordinate_system: CoordinateSystem, dtype=torch.float32):
        super().__init__()
        self.register_buffer("means", torch.tensor(means, dtype=dtype))
        self.register_buffer("stds", torch.tensor(stds, dtype=dtype))
        self.dim = self.means.shape[0]
        self.eps = 1e-8
        self.requires_nrml = False
        self.coordinate_system = coordinate_system

    def forward(self, num_samples):
        x = self.sample(num_samples)
        return x, self.log_prob(x)

    def sample(self, num_samples, normals = None):
        eps = torch.randn(num_samples, self.dim, device=self.means.device)
        return self.means + eps * self.stds
    
    def sample_with_pdf(self, num_samples, normals = None):
        eps = torch.randn(num_samples, self.dim, device=self.means.device)
        x = self.means + eps * self.stds
        pdf = self.log_prob(x, normals).exp()
        return x, pdf

    def prob(self, x, normals = None):
        # Add tiny epsions
        return self.log_prob(x, normals).exp()

    def log_prob(self, x, normals = None):
        diffs = (x - self.means) / self.stds

        return -0.5 * (
            diffs.pow(2).sum(dim=-1)
            + (2 * torch.log(self.stds)).sum() 
            + self.dim * math.log(2 * math.pi)
        )
    
    def requires_normal(self):
        return self.requires_nrml