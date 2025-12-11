"""
Create a base class that will hold a model and a way to sample from it, train it and visualize it 

"""


from cv2 import norm
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import tinycudann as tcnn

import normflows as nf
import normalizing_flows

from renderers import Renderer
import contexts
import utils as utils
from targets import Target

import mitsuba as mi
import drjit as dr

from coordinate_system import CoordinateSystem

class AbstractModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Abstract Model can't be used")

    def sample(self, num_samples: int):
        raise NotImplementedError("Abstract Model can't be used")
    
    def sample_with_pdf(self, num_samples: int):
        raise NotImplementedError("Abstract Model can't be used")
    
    def compute_batch_gradient(self, target, batch, prior_pdf, batch_pdf):
        raise NotImplementedError("Abstract Model can't be used")
    
    def visualize(self, target):
        raise NotImplementedError("Abstract Model can't be used")
    
    def RMSE(self, target, num_samples = 1000000):
        raise NotImplementedError("Abstract Model can't be used")
    

def batched_dot(z):
    return torch.bmm(z.unsqueeze(1), z.unsqueeze(2))[:, 0, 0]

def I(z):
    return torch.divide(z, (batched_dot(z) + 1).sqrt().unsqueeze(-1))

def I_abs_det(z):
    return 1 / (batched_dot(z) + 1).square()

###########################################################################################
### Normalizing Flow Model
###########################################################################################

class NormalizingFlowModel(AbstractModel):
    def __init__(self, 
                 prior, 
                 flows, 
                 context_size,
                 coordinate_system: CoordinateSystem,
                 scene_bbox,
                 has_normal_encoding,
                 has_incoming_encoding,
                 has_positional_encoding,
                 context_encoding_conf = None, 
                 domain = [[0, 1], [0, 1]]):
        super().__init__()
        if not has_normal_encoding and prior.requires_normal():
            raise RuntimeError("Conditioning mismatch: Prior requires normal conditioning but model does not, so it can't provide normal to prior.")

        self.prior = prior

        self.coordinate_system = coordinate_system

        self.has_normal_encoding = has_normal_encoding
        self.has_incoming_encoding = has_incoming_encoding
        self.has_positional_encoding = has_positional_encoding

        if context_encoding_conf is not None:
            self.context_encoding = tcnn.Encoding(context_size, context_encoding_conf)

        self.network = normalizing_flows.NormalizingFlow(flows)

        self.register_buffer("domain", torch.as_tensor(domain))
        bbox = torch.stack((scene_bbox.min.torch(), scene_bbox.max.torch()))
        self.register_buffer("scene_bbox", bbox)

    def from_domain(self, x, domain = None):
        """
        Transforms a sample from a given domain to [-1, 1]

        Args:
          x: Batch of samples
          domain: src domain, will use self.domain if None

        Returns:
          Batch in [-1, 1]
        """

        src_domain = domain if domain is not None else self.domain
        return (2 * x - src_domain[:, 0] - src_domain[:, 1]) / (src_domain[:, 1] - src_domain[:, 0])
    
    def to_domain(self, y, domain = None):
        """
        Transforms a sample from [-1, 1] to a given domain 

        Args:
          y: Batch of samples
          domain: target domain, will use self.domain if None

        Returns:
          Batch in domain
        """

        target_domain = domain if domain is not None else self.domain
        return (y + 1) / 2 * (target_domain[:, 1] - target_domain[:, 0]) + target_domain[:, 0]
    
    def normalize_pos_for_hashgrid(self, pos):
        """
        Normalize the positions into [0, 1] for hash grid encoding
        """
        return (pos - self.scene_bbox[0]) / (self.scene_bbox[1] - self.scene_bbox[0])



    def encode_context(self, normal = None, in_dir = None, position = None):
        device = self.domain.device
        normal_enc = torch.empty((0), device = device)
        in_dir_enc = torch.empty((0), device = device)
        pos_enc    = torch.empty((0), device = device)

        if self.has_normal_encoding:
            if normal is None:
                raise RuntimeError("model requires normal conditiong but no normals were provided")

            if len(normal.shape) == 1:
                normal = normal.unsqueeze(0)

            # normal_enc = normal
            normal_enc = (normal + 1) / 2

        if self.has_incoming_encoding:
            if in_dir is None:
                raise RuntimeError("model requires in direction conditiong but no incoming directions were provided")

            if len(in_dir.shape) == 1:
                in_dir = in_dir.unsqueeze(0)

            # in_dir_enc = in_dir
            in_dir_enc = (in_dir + 1) / 2

        if self.has_positional_encoding:
            if position is None:
                raise RuntimeError("model requires positional conditiong but no positions were provided")

            if len(position.shape) == 1:
                position = position.unsqueeze(0)

            pos_enc = self.normalize_pos_for_hashgrid(position)
        

        if hasattr(self, "context_encoding"):
            return self.context_encoding(torch.cat((normal_enc, in_dir_enc, pos_enc), dim = -1))
        
        return normal_enc # Return empty tensor, appending to model input will be no op
    

        # context_enc = torch.empty((0), device = device)
        # if hasattr(self, "context_encoding"):
        #     if normals is None and positions is None:
        #         raise RuntimeError("Model was given a context encoding on creation but no context on forward.")

        #     # At most one of these is None, create empty tensor to cat to non-None one
        #     if normals is None: normals = torch.empty((0), device = device)
        #     if positions is None: positions = torch.empty((0), device = device)

        #     context_enc = self.context_encoding(torch.cat((normals, positions), dim = -1))

        # return context_enc


    def forward(self, x, normal = None, in_dir = None, position = None):
        unsqueezed = False
        if len(x.shape) == 1:
            unsqueezed = True
            x = x.unsqueeze(0)

        context_enc = self.encode_context(normal, in_dir, position)

        x = self.from_domain(x)
        res = self.network.forward(x, context=context_enc)
        res = self.to_domain(res)

        if unsqueezed:
            return res.squeeze(0)
        return res
    
    def forward_and_logdet(self, x, normal = None, in_dir = None, position = None):
        unsqueezed = False
        if len(x.shape) == 1:
            unsqueezed = True
            x = x.unsqueeze(0)

        context_enc = self.encode_context(normal, in_dir, position)

        x = self.from_domain(x)
        res, log_det = self.network.forward_and_log_det(x, context=context_enc)
        res = self.to_domain(res)

        if unsqueezed:
            return res.squeeze(0)
        return res, log_det
    
    def log_prob(self, x, normal = None, in_dir = None, position = None, to_solid_angle = True):
        unsqueezed = False
        if len(x.shape) == 1:
            unsqueezed = True
            x = x.unsqueeze(0)

        context_enc = self.encode_context(normal, in_dir, position)

        z, abs_log_det = self.network.inverse_and_log_det(self.from_domain(x), context_enc)

        prior_log_prob = self.prior.log_prob(self.to_domain(z), normal)

        log_prob = prior_log_prob + abs_log_det

        if to_solid_angle:
            log_prob = log_prob - utils.system_to_cartesian_jacobian(x, self.coordinate_system).log()

        if unsqueezed:
            return log_prob.squeeze(0)
        return log_prob

    def log_prob_bsdf_prior(self, x, si: mi.SurfaceInteraction3f, normal = None, in_dir = None, position = None, to_solid_angle = True):
        unsqueezed = False
        if len(x.shape) == 1:
            unsqueezed = True
            x = x.unsqueeze(0)

        context_enc = self.encode_context(normal, in_dir, position)

        z, abs_log_det = self.network.inverse_and_log_det(self.from_domain(x), context_enc)

        world_dirs = utils.system_to_cartesian(z, self.coordinate_system)

        bsdf_ctx = mi.BSDFContext(mode = mi.TransportMode.Importance, type_mask=mi.BSDFFlags.Smooth)

        world_dirs_dr = mi.Vector3f(world_dirs.transpose(0, 1))
        bsdf_pdf = si.bsdf().pdf(bsdf_ctx, si, si.to_local(world_dirs_dr))

        log_bsdf_pdf = (bsdf_pdf.torch().detach() + 1e-5).log()

        if not to_solid_angle:
            log_bsdf_pdf = log_bsdf_pdf + (utils.system_to_cartesian_jacobian(z, self.coordinate_system) + 1e-5).log()

        if to_solid_angle:
            abs_log_det = abs_log_det - (utils.system_to_cartesian_jacobian(x, self.coordinate_system) + 1e-5).log()

        log_prob = log_bsdf_pdf + abs_log_det

        if unsqueezed:
            return log_prob.squeeze(0)
        return log_prob

    def sample(self, num_samples: int, normal = None, in_dir = None, position = None):
        normals_exp = normal
        if normal.shape[0] != num_samples:
            normals_exp = normal.expand(num_samples, -1)

        prior_samples = self.prior.sample(num_samples, normals_exp)   

        return self.forward(prior_samples, normal = normals_exp, in_dir = in_dir, position=position)
    
    def sample_with_pdf(self, num_samples: int, normal = None, in_dir = None, position = None, to_solid_angle = True):
        normals_exp = normal
        if normal.shape[0] != num_samples:
            normals_exp = normal.expand(num_samples, -1)

        prior_samples, pdf = self.prior.sample_with_pdf(num_samples, normals_exp)   
        samples, log_det = self.forward_and_logdet(prior_samples, normal=normals_exp, in_dir=in_dir, position=position)

        pdf /= torch.exp(log_det)

        if to_solid_angle:
            pdf = pdf / (utils.system_to_cartesian_jacobian(samples, self.coordinate_system) + 1e-6)

        return samples, pdf


    def sample_with_pdf_bsdf_prior(self, num_samples: int, si: mi.SurfaceInteraction3f, rng, normal = None, in_dir = None, position = None, to_solid_angle = True):
        normals_exp = normal
        if normal.shape[0] != num_samples:
            normals_exp = normal.expand(num_samples, -1)

        bsdf_ctx = mi.BSDFContext(mode = mi.TransportMode.Importance, type_mask=mi.BSDFFlags.Smooth)

        sample1 = rng.random(mi.Float, (num_samples, ))
        sample2 = rng.random(mi.Point2f, (2, num_samples))

        bsdf_sample, spec = si.bsdf().sample(bsdf_ctx, si, sample1, sample2)

        if dr.any(dr.all(spec == mi.Spectrum(0))):
            print(f"bsdf sampling failed")

        prior_samples = utils.cartesian_to_system(si.to_world(bsdf_sample.wo).torch().detach().transpose(0, 1), self.coordinate_system)

        # bsdf pdf to angle area measure
        pdf = bsdf_sample.pdf * utils.system_to_cartesian_jacobian(prior_samples, self.coordinate_system)

        samples, log_det = self.forward_and_logdet(prior_samples, normal=normals_exp, in_dir=in_dir, position=position)

        pdf /= torch.exp(log_det)

        if to_solid_angle:
            pdf = pdf / (utils.system_to_cartesian_jacobian(samples, self.coordinate_system) + 1e-6)

        return samples, pdf


    def compute_batch_gradient(self, 
                               batch, 
                               batch_pdf, 
                               bsdf_pdf, 
                               probs_p, 
                               si, 
                               rng, 
                               bsdf_prior: bool, 
                               mis_aware: bool, 
                               mis_beta: float, 
                               normal = None, 
                               in_dir = None, 
                               position = None):
        ### Compute f = Fp => F is constant factor of loss
        # probs_p = target_vals if target_vals is not None else target.prob(batch, normal)

        ### Compute model prediction on batch
        # log_probs_q = self.network.log_prob(batch, context = mr_grid_features)

        if bsdf_prior:
            log_probs_q = self.log_prob_bsdf_prior(batch, si = si, normal=normal, in_dir=in_dir, position=position)
        else:
            log_probs_q = self.log_prob(batch, normal=normal, in_dir=in_dir, position=position)

        log_target = log_probs_q
        if mis_aware:
            # MIS aware optimization by not learning distribution specifically where bsdf sampling is already good
            probs_q = log_probs_q.exp()
            q_weight = torch.pow(probs_q, mis_beta)
            p_weight = torch.pow(bsdf_pdf, mis_beta)
            c = q_weight / (q_weight + p_weight)

            log_target = (c * probs_q + (1 - c) * bsdf_pdf).log()

        ### Compute per-sample negative log likelihood
        nll = -probs_p * log_target / batch_pdf

        ### Pearson Chi squared divergence to minimize Variance
        # nll = -log_probs_q * (probs_p / batch_pdf) ** 2

        ###  reverse KLD -> doesn't seem to work
        # rKLD = log_probs_q.exp() / batch_pdf * (log_probs_q - probs_p.clamp(min=1e-6).log())
        # nll = rKLD

        ### Compute loss
        loss = torch.mean(nll[~(torch.isnan(nll) | torch.isinf(nll))])

        
        ### Backpropagate
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()

        return loss

    def visualize(self, 
                  renderer: Renderer, 
                  resolution = [100, 100], 
                  colormap = 'jet', 
                  context = None, 
                  bsdf_prior = False, 
                  only_direct = False, 
                  only_indirect = False,
                  tone_map_target = False):
        ### Build meshgrid
        xx, yy = torch.meshgrid(
            torch.linspace(self.domain[0][0], self.domain[0][1], resolution[0], ), 
            torch.linspace(self.domain[1][0], self.domain[1][1], resolution[1])
        )

        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.to(next(self.parameters()).device)


        si = renderer.sample_sensor_first_si()
        
        ### Evaluate target

        if context is None:
            context = contexts.Context(bounces = 4, 
                spp = 128,
                bsdf_sampling = True, 
                emitter_sampling = True,
                only_direct=only_direct,
                only_indirect=only_indirect
                )
        
        distribution = renderer.build_distribution_on_surface(si, 
            resolution=resolution, 
            coordinate_system=self.coordinate_system, 
            include_brdf=self.has_incoming_encoding,
            render_context=context)
        
        target_prob = utils.res_to_numpy_1(distribution, resolution[0], resolution[1])

        N = si.n.torch().transpose(0, 1)
        wi = si.to_world(si.wi).torch().transpose(0, 1)
        P = si.p.torch().transpose(0, 1)

        if bsdf_prior:
            log_prob = self.log_prob_bsdf_prior(zz, si, N, wi, P)
        else:
            log_prob = self.log_prob(zz, N, wi, P)
        prob = log_prob.exp()

        prob = prob.to('cpu').view(*xx.shape) 
        prob[torch.isnan(prob)] = 0
        prob = np.transpose(prob.detach().numpy(), (1, 0))

        ticks = np.linspace(0, 1, num = 5)
        x_ticks = ticks * resolution[0]
        y_ticks = ticks * resolution[1]
        tick_labels = utils.get_plt_ticks_from_coords(self.coordinate_system)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        if tone_map_target:
            target_prob = utils.tonemap(target_prob)

        ax1.imshow(target_prob, cmap=colormap) 
        ax1.set_title("Target")
        ax1.set_xticks(ticks=x_ticks, labels = tick_labels[0])
        ax1.set_yticks(ticks=y_ticks, labels = tick_labels[1])
        ax1.set_xlabel("Azimuthal angle φ")
        ax1.set_ylabel("Polar angle θ")

        ax2.imshow(prob, cmap=colormap)
        ax2.set_title("Model Prediction")
        ax2.set_xticks(ticks=x_ticks, labels = tick_labels[0])
        ax2.set_yticks(ticks=y_ticks, labels = tick_labels[1])
        ax2.set_xlabel("Azimuthal angle φ")
        ax2.set_ylabel("Polar angle θ")

        return fig



    def visualize_old(self, target: Target, grid_size = 100, colormap = 'jet', domain = None, normal = None, in_dir = None, position = None):
        ### Build meshgrid
        vis_domain = domain if domain is not None else self.domain
        
        xx, yy = torch.meshgrid(
            torch.linspace(vis_domain[0][0], vis_domain[0][1], grid_size), 
            torch.linspace(vis_domain[1][0], vis_domain[1][1], grid_size)
        )

        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.to(next(self.parameters()).device)



        if position is None and self.has_positional_encoding:
            if normal is None:
                print("Requires position and none was given: reverting to default position [0, 0, 0]")
                position = torch.tensor([0, 0, 0], device = self.domain.device, dtype=torch.float32)
            else:
                print("Requires position and none was given: reverting to position based on normal")
                position = normal.detach().clone()
            
        if normal is None and self.has_normal_encoding:
            print("Requires normal and none was given: reverting to default normal [0, 0, 1]")
            normal = torch.tensor([0, 0, 1], device = self.domain.device, dtype=torch.float32)

        if in_dir is None and self.has_incoming_encoding:
            print("Requires in direction and none was given: reverting to default in direction wi = N")
            in_dir = normal.detach().clone()

        

        ### Evaluate target
        prob = target.prob(zz, normal).to('cpu').view(*xx.shape)
        target_prob = np.flip(prob.detach().numpy(), axis = 1)

        log_prob = self.log_prob(zz, normal, in_dir, position)
        prob = log_prob.exp()

        try_measure_space_transform = False
        if try_measure_space_transform:
            sins = zz[..., 1].sin()
            
            prob = prob / torch.max(sins, torch.tensor([1e-4], device=zz.device))


        prob = prob.to('cpu').view(*xx.shape) 
        prob[torch.isnan(prob)] = 0
        prob = np.flip(prob.detach().numpy(), axis = 1)

        norm_probs = np.divide(prob, np.max(prob))
        norm_target = np.divide(target_prob, np.max(target_prob))

        diff_image = np.stack((norm_target, norm_probs, np.ones_like(prob)), axis = -1)
        diff_image = np.flip(np.transpose(diff_image, (1, 0, 2)), axis = 0)

        # prob = prob / prob.max()
        # target_prob = target_prob / target_prob.max()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # vmin = min(target_prob.min(), prob.min(), squared_error.min())
        # vmax = max(target_prob.max(), prob.max(), squared_error.max())

        # norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        ax1.pcolormesh(xx, yy, target_prob, cmap=colormap) 
        ax1.set_title("Target")

        ax2.pcolormesh(xx, yy, prob, cmap=colormap)
        ax2.set_title("Model Prediction")

        ax3.imshow(diff_image)
        ax3.set_title("Channelwise: R - target; G - model pred")

        # cf = ax3.pcolormesh(xx, yy, error, cmap=colormap)
        # ax3.set_title("Absolute Error")
        # fig.colorbar(cf, ax=ax3)
        
        # ax3.pcolormesh(xx, yy, prior * prob, cmap=colormap)
        # ax3.set_title("Prior modulated probs")
        
        
        return fig
    
    def RMSE(self, target, grid_size = 100, domain = None):
        vis_domain = domain if domain is not None else self.domain
        
        xx, yy = torch.meshgrid(
            torch.linspace(vis_domain[0][0], vis_domain[0][1], grid_size), 
            torch.linspace(vis_domain[1][0], vis_domain[1][1], grid_size)
        )

        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.to(next(self.parameters()).device)


        log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
        target_prob = np.flip(torch.exp(log_prob).numpy(), axis = 1)


        # Samples need to be in [-1, 1] for the model
        zz = self.from_domain(zz, domain)

        log_prob = self.network.log_prob(zz).to('cpu').view(*xx.shape)
        model_prob = torch.exp(log_prob)
        model_prob[torch.isnan(model_prob)] = 0
        model_prob = np.flip(model_prob.numpy(), axis = 1)
        
        return np.square(model_prob - target_prob)


