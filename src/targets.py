import torch
import mitsuba as mi
from coordinate_system import CoordinateSystem
import utils as utils

class Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def log_prob(self, z):
        raise NotImplementedError("Abstract target does not have log probability")
    
    def prob(self, z):
        raise NotImplementedError("Abstract target does not have log probability")



if mi.variant() == None:
    print("Loading targets without having set a mitsuba variant: RendererTarget won't be available")
    print("Call mi.set_variant() to set one or see https://mitsuba.readthedocs.io/en/stable/")
else:
    import renderers as rr 
    import kernels
    import drjit as dr

    class RendererTarget(Target):
        def __init__(self, renderer: rr.Renderer, coordinate_system: CoordinateSystem):
            self.renderer = renderer
            self.coordinate_system = coordinate_system

        def log_prob(self, z, normals = None):
            return self.prob(z, normals).log()

        def prob(self, z, normals = None):
            samples = mi.Point2f(z.transpose(0, 1))

            # dirs = kernels.angles_to_dirs(samples[:, 0], samples[:, 1])
            dirs = kernels.system_to_cartesian(samples, self.coordinate_system)
            locs = dirs * 10

            spec = self.renderer.eval_samples(samples, locs, self.coordinate_system)
            if normals is not None:
                if normals.shape[0] != dr.width(samples):
                    normals = normals.expand(dr.width(samples), -1)

                spec = kernels.modulate_with_cosine(samples, spec, mi.Vector3f(normals.transpose(0, 1)))

            return spec.torch().transpose(0, 1).mean(dim = -1)

    

class UniformTarget(Target):
    def __init__(self, domain, coordinate_system: CoordinateSystem):
        super().__init__()
        self.register_buffer("domain", torch.as_tensor(domain))

        self.pdf = 1 / (domain[:, 1] - domain[:, 0]).prod(dim = 0)
        self.log_pdf = torch.log(self.pdf)
        self.coordinate_system = coordinate_system

    def in_domain_mask(self, z):
        return  ((z >= self.domain[:, 0]) & (z <= self.domain[:, 1])).all(dim=1)

    def log_prob(self, z, normal = None):
        return torch.where(self.in_domain_mask(z), self.log_pdf, 0)
    
    def prob(self, z, normal = None):
        return torch.where(self.in_domain_mask(z), self.pdf, 0)


# Class taken and adapted from the source code of https://github.com/VincentStimper/normalizing-flows
class ImageTarget(Target):
    """
    Intensities of an image determine probability density of prior
    """

    def __init__(self, 
                 image, 
                 coordinate_system: CoordinateSystem,
                 domain = [[0, 1], [0, 1]],
                 eps=1.0e-10, 
                 dtype=torch.float32, 
                 normalize = False):
        """Constructor

        Args:
          image: image as np matrix
          index_0_1: Whether to index from [0, 1]^2 instead of [-1, 1]^2
          eps: small value to add to image to avoid log(0) problems
        """
        super().__init__()

        image_tens = torch.tensor(image, dtype=dtype)

        self.coordinate_system = coordinate_system

        if normalize:
            image_ = (image_tens / image_tens.sum() + eps)[None, None, ..., ...]
        else:
            image_ = image_tens[None, None, ..., ...]
        self.image_dims = image_.shape
        
        self.register_buffer("domain", torch.as_tensor(domain))
        self.register_buffer("log_eps", torch.log(torch.tensor(eps, dtype=dtype)))
        self.register_buffer(
            "density", image_
        )

    def from_domain(self, x):
        x = (x - self.domain[:, 0]) / (self.domain[:, 1] - self.domain[:, 0])
        return x * 2 - 1
    
    def to_domain(self, x):
        x = (x + 1) / 2
        return x * (self.domain[:, 1] - self.domain[:, 0])
        

    def log_prob(self, z, normals = None):
        """
        Args:
          z: value or batch of latent variable
          normals: normals, as 3D vectors

        Returns:
          log probability of the distribution for z
        """

        # Map into [-1, 1] when user wants to index in [0, 1]
        z_norm = self.from_domain(z)

        z_expanded = z_norm[None, None, ..., ...]

        log_probs = torch.log(torch.nn.functional.grid_sample(self.density, z_expanded, mode='nearest').view(-1))

        if normals is not None:
            if normals.shape[0] != z.shape[0]:
                    normals = normals.expand(z.shape[0], 0, normals.shape[0])

            # vecs = utils.to_vec3_torch(z[..., 0], z[..., 1])
            vecs = utils.system_to_cartesian(z, self.coordinate_system)

            sum = normals * vecs
            dots = torch.clamp(torch.sum(sum, dim = 1), min = 0)

            log_probs = log_probs + torch.log(dots)

        return torch.where(torch.isinf(log_probs), -self.log_eps, log_probs)
    
    def prob(self, z, normals = None):
        """
        Args:
          z: value or batch of latent variable
          normals: normals, as 3D vectors

        Returns:
          probability of the distribution for z
        """

        # Map into [-1, 1] when user wants to index in [0, 1]
        z_norm = self.from_domain(z)

        z_expanded = z_norm[None, None, ..., ...]

        probs = torch.nn.functional.grid_sample(self.density, z_expanded).view(-1)

        if normals is not None:
            # vecs = utils.to_vec3_torch(z[..., 0], z[..., 1])
            vecs = utils.system_to_cartesian(z, self.coordinate_system)

            sum = normals * vecs
            dots = torch.clamp(torch.sum(sum, dim = 1), min = 0)

            probs = probs * dots

        return probs


