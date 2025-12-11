import numpy as np
from torch import nn
import torch
import normflows as nf
from normflows.flows.base import Flow
import matplotlib.pyplot as plt
import utils as utils


def compare_model(nfm, device, grid_size = 100, low = -3.0, high = 3.0):
    # Plot prior distribution
    xx, yy = torch.meshgrid(torch.linspace(low, high, grid_size), torch.linspace(low, high, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.double().to(device)
    log_prob = nfm.p.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ax1.pcolormesh(xx, yy, prob.data.numpy())
    ax1.pcolormesh(xx, yy, prob.data.numpy())

    # Plot initial posterior distribution
    log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax2.pcolormesh(xx, yy, prob.data.numpy())
    plt.show()

def compare_model_normal(nfm, device, normal, grid_size = 100):
    # Plot prior distribution
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.double().to(device)

    normals = normal.expand((*zz.shape[:-1], ) + (normal.shape[0], ))


    log_prob = nfm.p.log_prob(zz, normal).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ax1.pcolormesh(xx, yy, prob.data.numpy())
    ax1.pcolormesh(xx, yy, prob.data.numpy())

    # Plot initial posterior distribution
    log_prob = nfm.log_prob(zz, normals).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax2.pcolormesh(xx, yy, prob.data.numpy())
    plt.show()

def compare_model_context(nfm, device, mrhEncoding, grid_size = 100):
    # Plot prior distribution
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.double().to(device)

    log_prob = nfm.p.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ax1.pcolormesh(xx, yy, prob.data.numpy())
    ax1.pcolormesh(xx, yy, prob.data.numpy())

    context = mrhEncoding(zz)
    # Plot initial posterior distribution
    log_prob = nfm.log_prob(zz, context).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax2.pcolormesh(xx, yy, prob.data.numpy())
    plt.show()

def compare_model_context_normal(nfm, device, normal, mrhEncoding, grid_size = 100, dtype = torch.float32):
    # Plot prior distribution
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.type(dtype=dtype).to(device)

    log_prob = nfm.p.log_prob(zz, normal).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ax1.pcolormesh(xx, yy, prob.data.numpy())
    ax1.pcolormesh(xx, yy, prob.data.numpy())

    context = mrhEncoding(zz)
    if normal is None:
        normal = torch.tensor([0, 0, 0], dtype=dtype).to(device)
    normals = normal.expand((*zz.shape[:-1], ) + (normal.shape[0], ))


    model_input = torch.cat((normals, context), dim = 1)

    # Plot initial posterior distribution
    log_prob = nfm.log_prob(zz, model_input).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax2.pcolormesh(xx, yy, prob.data.numpy())
    plt.show()

def compare_model_context_normal_enc(nfm, device, normal, mrhEncoding_normal, mrhEncoding, grid_size = 100):
    # Plot prior distribution
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.double().to(device)

    normals = normal.expand((*zz.shape[:-1], ) + (normal.shape[0], ))

    log_prob = nfm.p.log_prob(zz, normal).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ax1.pcolormesh(xx, yy, prob.data.numpy())
    ax1.pcolormesh(xx, yy, prob.data.numpy())

    context = mrhEncoding(zz)
    normal_enc = mrhEncoding_normal(normal.unsqueeze(0)).squeeze(0)

    normals_enc = normal_enc.expand((*zz.shape[: -1], ) + (normal_enc.shape[0], ))

    model_input = torch.cat((normals_enc, context), dim = 1)

    # Plot initial posterior distribution
    log_prob = nfm.log_prob(zz, model_input).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax2.pcolormesh(xx, yy, prob.data.numpy())
    plt.show()


class SimpleAffine(Flow):
    def __init__(self, dims = 2):
        super().__init__()
        self.dims = dims
        self.a = nn.Parameter(torch.zeros(self.dims))
        self.b = nn.Parameter(torch.zeros(self.dims))

    def forward(self, x):
        y = torch.exp(self.a) * x + self.b

        # calculate jacobian determinant in log space
        # log and exp term cancel, leaving only sum
        log_jacobian_det = torch.ones(y.shape[0]) * self.a.sum()

        return y, log_jacobian_det
    
    def inverse(self, y):
        x = (y - self.b) / torch.exp(self.a)
    
        # calculate inverse log determinant in log space
        # log and exp cancel, see above, inverse is transformed to negative by log before cancelling exp
        inv_log_jacobian_det = torch.ones(x.shape[0]) * -self.a.sum()

        return x, inv_log_jacobian_det


class RealNVPFlow(Flow):
    def __init__(self, mask, hidden_size):
        super().__init__()
        self.dim = len(mask)
        # self.mask = nn.Parameter(mask, requires_grad=False)
        self.mask = torch.Tensor(mask)


        self.s_func = nn.Sequential(nn.Linear(in_features=self.dim, out_features=hidden_size, dtype=torch.float64), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=torch.float64), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=self.dim, dtype=torch.float64))

        self.scale = nn.Parameter(torch.ones(self.dim))

        self.t_func = nn.Sequential(nn.Linear(in_features=self.dim, out_features=hidden_size, dtype=torch.float64), nn.LeakyReLU(),
                                     nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=torch.float64), nn.LeakyReLU(),
                                     nn.Linear(in_features=hidden_size, out_features=self.dim, dtype=torch.float64))

    def forward(self, x):
        x_mask = x*self.mask
        s = self.s_func(x_mask) * self.scale
        t = self.t_func(x_mask)

        y = x_mask + (1 - self.mask) * (x * torch.exp(s) + t)
        # Sum for -1, since for every batch, and 1-mask, since the log_det_jac is 1 for y1:d = x1:d.
        log_det_jac = ((1 - self.mask) * s).sum(-1)
        
        return y, log_det_jac

    def inverse(self, y):
        y_mask = y * self.mask

        s = self.s_func(y_mask) * self.scale
        t = self.t_func(y_mask)

        x = y_mask + (1 - self.mask)*(y - t) * torch.exp(-s)
        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)

        return x, inv_log_det_jac



class LambertConditionedImagePrior(nf.distributions.ImagePrior):
    """
    Initial Implementation taken from ImagePrior class in https://github.com/VincentStimper/normalizing-flows
    """

    def __init__(self, image, x_range=[-3, 3], y_range=[-3, 3], eps=1.0e-10):
        super().__init__(image, x_range = x_range, y_range = y_range, eps = eps)
        self.iterations_count = [0] * 101

    def log_prob(self, z, normals = None):
        z_ = torch.clamp((z - self.shift) / self.scale, max=1, min=0)
        ind = (z_ * (self.image_size - 1)).long()

        if normals == None:
            return self.density[ind[:, 0], ind[:, 1]]


        vecs = utils.points_to_unit_sphere(z_)

        sum = normals * vecs
        dots = torch.clamp(torch.sum(sum, dim = 1), min = 0)

        return self.density[ind[:, 0], ind[:, 1]] + torch.log(dots)
    

    def rejection_sampling(self, normals = None, num_steps=1):
        """Perform rejection sampling on image distribution

        Args:
         num_steps: Number of rejection sampling steps to perform

        Returns:
          Accepted samples
        """
        z_ = torch.rand(
            (num_steps, 2), dtype=self.image.dtype, device=self.image.device
        )
        prob = torch.rand(num_steps, dtype=self.image.dtype, device=self.image.device)
        ind = (z_ * (self.image_size - 1)).long()
        intensity = self.image[ind[:, 0], ind[:, 1]]

        if normals == None:
            accept = intensity > prob
            z = z_[accept, :] * self.scale + self.shift
            return z

        vecs = utils.points_to_unit_sphere(z_)
        sum = normals * vecs
        dots = torch.clamp(torch.sum(sum, dim = 1), min = 0)

        accept = dots * intensity > prob
        z = z_[accept, :] * self.scale + self.shift
        return z


    def sample(self, normals = None, num_samples=1):
        """Sample from image distribution through rejection sampling

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples
        """
        z = torch.ones((0, 2), dtype=self.image.dtype, device=self.image.device)
        count = 0
        while len(z) < num_samples and count < 100:
            count += 1
            z_ = self.rejection_sampling(normals, num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind, :]], 0)
        self.iterations_count[count] += 1
        return z    



class Uniform(nf.distributions.BaseDistribution):
    """
    Multivariate uniform distribution
    """

    def __init__(self, shape, low=-1.0, high=1.0, device = "cpu"):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          low: Lower bound of uniform distribution
          high: Upper bound of uniform distribution
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.d = np.prod(shape)
        low = torch.tensor(low).detach()
        high = torch.tensor(high).detach()
        self.log_prob_val = torch.tensor(-self.d * np.log(high - low)).to(device).detach()
        self.low = low.to(device)
        self.high = high.to(device)

    def forward(self, num_samples=1, context=None):
        eps = torch.rand(
            (num_samples,) + self.shape, dtype=self.low.dtype, device=self.low.device
        )
        z = self.low + (self.high - self.low) * eps
        log_p = self.log_prob_val * torch.ones(num_samples, device=self.low.device)
        return z, log_p

    def log_prob(self, z, context=None):
        log_p = self.log_prob_val * torch.ones(z.shape[0], device=z.device)
        out_range = torch.logical_or(z < self.low, z > self.high)
        ind_inf = torch.any(torch.reshape(out_range, (z.shape[0], -1)), dim=-1)
        log_p[ind_inf] = -np.inf
        return log_p





# currently not encoded
class ConditionedRVPFlow(Flow):
    def __init__(self, mask, hidden_size, context_size):
        super().__init__()
        self.dim = len(mask)
        self.context_size = context_size
        self.mask = torch.Tensor(mask)

        self.input_size = self.dim + self.context_size

        self.s_func = nn.Sequential(nn.Linear(in_features=self.input_size, out_features=hidden_size, dtype=torch.float64), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=torch.float64), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=self.dim, dtype=torch.float64))

        self.scale = nn.Parameter(torch.ones(self.dim))

        self.t_func = nn.Sequential(nn.Linear(in_features=self.input_size, out_features=hidden_size, dtype=torch.float64), nn.LeakyReLU(),
                                     nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=torch.float64), nn.LeakyReLU(),
                                     nn.Linear(in_features=hidden_size, out_features=self.dim, dtype=torch.float64))

    def forward(self, x, context):
        x_mask = x * self.mask

        context_vec = context.expand((*x_mask.shape[:-1], ) + (context.shape[0], ))

        inp = torch.cat((x_mask, context_vec), dim = len(x_mask.shape) - 1)
        s = self.s_func(inp) * self.scale
        t = self.t_func(inp)

        y = x_mask + (1 - self.mask) * (x * torch.exp(s) + t)
        # Sum for -1, since for every batch, and 1-mask, since the log_det_jac is 1 for y1:d = x1:d.
        log_det_jac = ((1 - self.mask) * s).sum(-1)
        
        return y, log_det_jac

    def inverse(self, y, context):
        y_mask = y * self.mask

        context_vec = context.expand((*y_mask.shape[:-1], ) + (context.shape[0], ))

        inp = torch.cat((y_mask, context_vec), dim = len(y_mask.shape) - 1)
        s = self.s_func(inp) * self.scale
        t = self.t_func(inp)

        x = y_mask + (1 - self.mask)*(y - t) * torch.exp(-s)
        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)

        return x, inv_log_det_jac



class ConditionedNormalizingFlow(nn.Module):
    """
    Initial Implementation taken from NormalizingFlow class in https://github.com/VincentStimper/normalizing-flows
    """

    def __init__(self, q0, flows, p=None):
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p


    def forward(self, x, context):
        log_det = torch.zeros(len(x), device=x.device)
        for flow in self.flows:
            x, log_d = flow(x, context)
            log_det += log_d
        return x, log_det


    def log_prob(self, y, context):
        log_q = torch.zeros(len(y), dtype=y.dtype, device=y.device)
        z = y
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

# class PieceWiseLinear(Flow):
#     """
#     Implemented from Neural Importance Samöling, Müller et. al (https://arxiv.org/abs/1808.03856)
#
#     Implementation seems very difficult to get working nicely with batching etc.
#       Probably possible, but wont keep wasting time on it right now
#     """
# 
#     def __init__(self, mask, K, hidden_size):
#         super().__init__()
#         self.K = K
#         self.w = 1 / self.K
#         self.dim = len(mask)
#         self.mask = torch.Tensor(mask)
#         self.inv_mask = torch.Tensor(1 - mask)
#         
#         self.m = nn.Sequential(nn.Linear(in_features=self.dim, out_features=hidden_size, dtype=torch.float64), nn.LeakyReLU(),
#                                      nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=torch.float64), nn.LeakyReLU(),
#                                      nn.Linear(in_features=hidden_size, out_features=self.dim, dtype=torch.float64), nn.Softmax())
# 
#     def forward_single(self, x):
#         x_masked = self.mask * x
#         Q = self.m(x_masked)
# 
#         b = torch.floor(self.K * x)
#         alpha = self.K * x - b
#         
#         sum = Q[:b].sum() + self.alpha * Q[b]
# 
#         sum = self.inv_mask * (self.inv_mask * sum + self.inv_mask * alpha)
#         y = x_masked + sum
# 
#         jacobian = Q[b] / self.w
# 
#         return y, torch.log(jacobian)
# 
#     def forward(self, x):
#         y = torch.zeros_like(x)
#         log_jacs = torch.zeros_like(x)
#         for index, el in enumerate(x):
#             y[index], log_jacs[index] = self.forward_single(el)
# 
#         return y, log_jacs
# 
#     def inverse_single(self, y):
#         y_fixed = self.mask * y
#         Q = self.m(y_fixed)
# 
#         lower_sum = 0
#         sum = Q[0]
#         b = 0
#         y_unfixed = self.inv_mask * y
#         while torch.all(sum < y_unfixed):
#             b += 1
#             lower_sum = sum
#             sum += Q[b]
# 
#         x = (self.inv_mask * y - lower_sum + Q[b] * b) / Q[b] * self.K
#         inv_jacobian = Q[b] / self.w
# 
#         return x, torch.log(inv_jacobian)
# 
# 
#     def inverse(self, y):
#         x = torch.zeros_like(y)
#         inv_log_jacs = torch.zeros_like(y)
#         for index, el in enumerate(y):
#             x[index], inv_log_jacs[index] = self.forward_single(el)
# 
#         return y, inv_log_jacs
