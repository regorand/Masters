from numpy import apply_along_axis
import torch
import math
import utils as utils
from coordinate_system import CoordinateSystem

class Prior(torch.nn.Module):
    def sample(self, num_samples, normals = None):
        raise NotImplementedError("Not Implemented")
    
    def prob(self, x, normals = None):
        raise NotImplementedError("Not Implemented")

    def log_prob(self, x, normals = None):
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

    def sample(self, num_samples, normals = None):
        return torch.rand((num_samples, self.dim), device =self.domain.device) * (self.domain[:, 1] - self.domain[:, 0]) + self.domain[:, 0]
    
    def sample_with_pdf(self, num_samples, normals = None, solid_angle = False):
        samples = self.sample(num_samples=num_samples, normals=normals)
        pdf = torch.full((num_samples,), self.pdf, device = self.domain.device)

        if not solid_angle:
            pdf * samples[:, 1].sin()

        return samples, pdf


    def prob(self, x, normals = None, solid_angle = False):
        if solid_angle:
            torch.full((x.shape[0], ), self.pdf, device = self.domain.device)

        return torch.full((x.shape[0], ), self.pdf, device = self.domain.device) * x[:, 1].sin()

    def log_prob(self, x, normals = None, solid_angle = False):
        return (self.prob(x, normals, solid_angle=solid_angle) + self.eps).log()

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

        # Pick helper vectors per batch entry
        # If |Nx| > |Nz| → use (−Ny, Nx, 0)
        # Else → use (0, −Nz, Ny)
        h = torch.where(
            cond.unsqueeze(-1),
            torch.stack([-N[:, 1], N[:, 0], torch.zeros_like(N[:, 0])], dim=-1),
            torch.stack([torch.zeros_like(N[:, 0]), -N[:, 2], N[:, 1]], dim=-1)
        )

        # Tangent T = normalize(h)
        T = torch.nn.functional.normalize(h, dim=-1)
        # Bitangent B = N × T
        B = torch.cross(N, T, dim=-1)

        # Stack columns [T, B, N] into rotation matrix
        # Shape: [B, 3, 3]
        onb = torch.stack((T, N, B), dim=-1)
        return onb


    def apply_ONB(self, samples, onb):
        """
        Rotate local (hemisphere) directions to world using ONB matrices.
        Args:
            local_dirs: [B, 3] tensor of local directions (hemisphere about +Z)
            onb: [B, 3, 3] ONB matrices from build_onb
        Returns:
            world_dirs: [B, 3] tensor of rotated directions
        """
        # matrix multiply: each batch entry gets its own ONB @ dir
        # (B, 3, 3) @ (B, 3, 1) → (B, 3, 1)
        world_dirs = (onb @ samples.unsqueeze(-1)).squeeze(-1)
        # normalize (optional, to clean up floating-point drift)
        world_dirs = torch.nn.functional.normalize(world_dirs, dim=-1)
        return world_dirs
    

    def sample(self, num_samples, normals):
        samples = self.sample_local(num_samples, normals.device)
        onb = self.build_ONB(normals)
        rotated_samples = self.apply_ONB(samples, onb)

        return utils.cartesian_to_system(rotated_samples, self.coordinate_system)
    
    def sample_with_pdf(self, num_samples, normals, solid_angle = False):
        samples = self.sample_local(num_samples, normals.device)
        onb = self.build_ONB(normals)
        rotated_samples = self.apply_ONB(samples, onb)

        # phi, theta = utils.to_angles_torch(rotated_samples)
        angles = utils.cartesian_to_system(rotated_samples, self.coordinate_system)
        dot = (rotated_samples * normals).sum(dim=-1)

        pdf = torch.clamp(dot, min = 0.0) / torch.pi
        
        if not solid_angle:
            pdf = pdf * angles[:, 1].sin()

        return angles, pdf
    
    def prob(self, x, normals, solid_angle = False):
        # vecs = utils.to_vec3_torch(x[:, 0], x[:, 1])
        vecs = utils.system_to_cartesian(x, self.coordinate_system)
        dot = (vecs * normals).sum(dim=-1)
        pdf = torch.clamp(dot, min = 0.0) / torch.pi
        if solid_angle:
            return pdf
        
        return pdf * torch.sin(x[:, 1])

    def log_prob(self, x, normals, solid_angle = False):
        return (self.prob(x, normals, solid_angle=solid_angle) + self.eps).log()

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