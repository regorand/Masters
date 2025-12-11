"""
This file is meant for code that I don't think I am gonna be using again, but I want to keep around just in case

I will mostly copy it into here with no special regard for it to be runnable from here

"""

import numpy as np
import torch
import ipywidgets as widgets

from scipy.special import rel_entr

def dist_abs_dist(dist1, dist2):
    return np.abs(np.subtract(dist1.astype(np.float64), dist2.astype(np.float64))).astype(np.uint8)

def KL_divergence_as_array(dist1, dist2):
    if dist1.shape != dist2.shape:
        print("Error, trying to compare dists with different shapes")
        return

    dist1 = np.divide(dist1, np.sum(dist1), casting="unsafe")
    dist2 = np.divide(dist2, np.sum(dist2), casting="unsafe")
    
    return rel_entr(dist1, dist2)

def KL_divergence(dist1, dist2):
    return np.ma.masked_invalid(KL_divergence_as_array(dist1, dist2)).sum()

def squared_dist_as_array(dist1, dist2, normalize = True):
    if dist1.shape != dist2.shape:
        print("Error, trying to compare dists with different shapes")
        return
    count = 1
    if normalize:
        for d in dist1.shape:
            count *= d
    return np.square(np.subtract(dist1, dist2)) / count

def squared_dist(dist1, dist2, normalize = True):
    if dist1.shape != dist2.shape:
        print("Error, trying to compare dists with different shapes")
        return
    return np.sum(squared_dist_as_array(dist1, dist2, normalize))





def evaluate_with_jacobian(model, input_points, inference_steps, requires_grad = False):
    model.eval().requires_grad_(requires_grad)

    jacobians = []

    for i, t in enumerate(torch.linspace(0, 1, inference_steps), start=1):
        t_values = t.expand(input_points.size(0))
        jacobians.append(torch.autograd.functional.jacobian(model, (input_points, t_values)))
        pred = model(input_points, t_values)
        input_points = input_points + (1 / inference_steps) * pred
        input_points = model.domain.wrap_torch(input_points)

    jacob_det = 1
    for j in reversed(jacobians):
        # wrong!
        a = j[0][0][0][0][0]
        b = j[0][0][0][0][1]
        c = j[0][0][1][0][0]
        d = j[0][0][1][0][1]

        jacob_det /= abs(a * d - b * c)


    return input_points, jacob_det




## TODO Wrapping needed here ?
def train_student_model(trainer_model, inference_steps, context: Context):
    student_model = model.MLP(layers=2, channels=64)

    optim = torch.optim.AdamW(student_model.parameters(), lr=context.lr)

    pbar = tqdm.tqdm(range(context.training_steps))
    losses = []
    for i in pbar:
        # indices = torch.randint(data.size(0), (hyper_params.batch_size,))
        # x1 = data[indices]
        x0 = torch.Tensor(utils.sample_uniform_points(context.batch_size))
        x1 = trainer_model.evaluate(x0, inference_steps, True)
    
        target = x1 - x0
    
        t = torch.rand(x1.size(0))
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        pred = student_model(xt, t)
        loss = ((target - pred) ** 2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())

    return student_model




def test_model(model):
    torch.manual_seed(42)
    model.eval().requires_grad_(False)
    xt = torch.rand(1000, 2)

    saved_points = []

    steps = 10
    plot_every = 2
    for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
        pred = model(xt, t.expand(xt.size(0)))
        xt = xt + (1 / steps) * pred
        xt = model.domain.wrap_torch(xt)
    
        if i % plot_every == 0:
            saved_points.append(xt)


    @widgets.interact(step=(0, int(steps / plot_every) - 1, 1))
    def update(step = 1):
    # [l.remove() for l in ax.lines]
        fig, ax = plt.subplots(figsize=(6, 6))

        # ax.scatter(sampled_points[:, 0], sampled_points[:, 1], color="black", marker="o")
        ax.scatter(saved_points[step][:, 0], saved_points[step][:, 1], color="blue", marker="o")

def display_dist_images(context: Context, models, overlay = False):
    for model in models:
        dist_image = 255 * dist_grid_visualization(model[1], context, 10, 10).astype(np.uint8)
        name = model[0]
        if overlay:
            cv2.imshow(name, np.clip(context.target_dist.dist.astype(np.uint16) + dist_image.astype(np.uint16), 0, 255).astype(np.uint8))
        else: 
            cv2.imshow(name, dist_image)

    cv2.imshow("actual distribution", context.target_dist.dist)
    
    cv2.waitKey()
    cv2.destroyAllWindows()


        # bsdf_sample, bsdf_spec = bsdf.sample(bsdf_ctx, 
        #                                      si, 
        #                                      rng.random(mi.Float, (dr.width(active))), 
        #                                      rng.random(mi.Point2f, (2, dr.width(active))), 
        #                                      active)

        # bsdf_pdf_weight = dr.square(bsdf_sample.pdf)
        # model_pdf_weight = dr.square(model_pdf)

        # Since we only have two weights and they are normalized to [0, 1], as are the random vars
        # Only need to fully compute one
        # bsdf_pdf_weight /= (bsdf_pdf_weight + model_pdf_weight)

        # choices = rng.random(mi.Float, dr.width(active))
        # use_bsdf = choices <= bsdf_pdf_weight

        # new_dir = dr.select(use_bsdf, si.to_world(bsdf_sample.wo), new_dir_model)
        # spec    = dr.select(use_bsdf, bsdf_spec, bsdf.eval(bsdf_ctx, si, si.to_local(new_dir), active=active))
        # pdf     = dr.select(use_bsdf, bsdf_sample.pdf, model_pdf)



###########################################################################################
### Reparam Flow Model
###########################################################################################


class Layer(nn.Module):
    def __init__(self, in_features, out_features, activation = nn.ReLU()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin_forward = nn.Linear(self.in_features, self.out_features)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.lin_forward(x))

class MLP(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_layers: list[int], 
                 activation = nn.ReLU(),
                 out_activation = nn.Identity()):
        super().__init__()
        if len(hidden_layers) < 0:
            raise Exception("MLP requires at least one hidden layer")
                        
        layers = [in_features] + hidden_layers
        self.net = nn.Sequential(*[
            Layer(layer_in, layer_out, activation) for layer_in, layer_out in zip(layers, layers[1:])
        ])

        self.out_proj = Layer(in_features=hidden_layers[-1], out_features=out_features, activation=out_activation)

    def forward(self, x):
        return self.out_proj(self.net(x))
    
    def input_shape(self):
        return self.net[0].channels_in




# TODO this belongs in own category of post processor function in network
def wrap(input, mod_val): 
    return torch.fmod(torch.fmod(input, mod_val) + mod_val, mod_val)


class ReparameterizationModel(AbstractModel):
    def __init__(self, features, prior, encoding, layers, domain = [[0, 1], [0, 1]]):
        super().__init__()
        self.prior = prior
        self.encoding = encoding
        self.layers = layers
        encoding_size = encoding.n_output_dims if type(encoding) == tcnn.Encoding else encoding.encoding_size
        self.network = MLP(encoding_size, features, layers, nn.SiLU())
        self.jac_func = torch.func.vmap(torch.func.jacrev(self, chunk_size=1))

        domain = torch.as_tensor(domain)

        self.register_buffer("domain", domain)

        self.alpha = 1e-3

    def forward(self, x: torch.Tensor):
        unsqueezed = False
        if len(x.shape) == 1:
            unsqueezed = True
            x = x.unsqueeze(0)

        # renormalize domain into [0, 1], everything outside will hash index into some table position, but it should be unlikely
        x = (x - self.domain[:, 0]) / (self.domain[:, 1] - self.domain[:, 0])

        enc = self.encoding(x)
        res = x + self.network(enc)
        
        res = self.domain[:, 0] + res * (self.domain[:, 1] - self.domain[:, 0])
        # res = wrap(res, self.domain[:, 1])

        if unsqueezed:
            return res.squeeze(0)
        return res
    
    def sample(self, num_samples: int):
        z = self.prior.sample(num_samples)
        return self(z)
    
    def compute_batch_gradient(self, target, batch, prior_pdf, batch_pdf):   
        jac_func = torch.func.vmap(torch.func.jacrev(self, chunk_size=1))    
         
        # Compute jacobians of transform
        jacobians = jac_func(batch)
        abs_dets = torch.det(jacobians).clamp(min = 1e-4)

        # Compute f(T)
        y = self(batch)
        probs_f = target.log_prob(y).exp()
        # probs_f = torch.divide(probs_f, torch.mean(probs_f / batch_pdf))

        # Compute f(I) and determinants
        Is = I(batch)
        probs_I = target.log_prob(Is).exp()
        dets_I = I_abs_det(batch)

        domain_loss_factor = 0  
        domain_loss = torch.maximum(torch.maximum(self.domain[:, 0] - y, y - self.domain[:, 1]), torch.zeros_like(y)).sum(-1)

        # Compute loss
        f_loss = (1 - self.alpha) * probs_f * abs_dets
        I_loss =      self.alpha  * probs_I * dets_I

        losses = -torch.log(f_loss + I_loss) * prior_pdf / batch_pdf + domain_loss_factor * domain_loss
        loss = torch.mean(losses[~(torch.isnan(losses) | torch.isinf(losses))])
        
        # Backpropagation
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()

        return loss

    def visualize(self, target, grid_size = 100, num_samples = 100000, colormap = 'jet', samples_in_domain = True):
        xx, yy = torch.meshgrid(
            torch.linspace(self.domain[0][0], self.domain[0][1], grid_size), 
            torch.linspace(self.domain[1][0], self.domain[1][1], grid_size)
        )
        
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.to(next(self.parameters()).device)

        log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
        prob = torch.exp(log_prob)


        q=self.prior.log_prob(zz).exp().cpu().view(*xx.shape)
        

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        ax1.pcolormesh(xx, yy, prob.cpu().detach(), cmap=colormap)
        ax1.set_title("target")
        
        ax2.pcolormesh(xx, yy, q, cmap=colormap)
        ax2.set_title("prior")
                
        x = self.prior.sample(num_samples)
        pred_samples = self(x).cpu().detach().numpy()
        if samples_in_domain:
            ax3.hist2d(pred_samples[:, 0], pred_samples[:, 1], bins=100, range=self.domain.cpu(), cmap=colormap, norm="linear")
        else:
            ax3.hist2d(pred_samples[:, 0], pred_samples[:, 1], bins=100, cmap=colormap, norm="linear")
        ax3.set_title("drawn samples")

        return fig
    
    def RMSE(self, target, num_samples = 1000000, bins = 50):
        x = self.prior.sample(num_samples)
        pred_samples = self(x).cpu().detach().numpy()

        H, _, _ = np.histogram2d(pred_samples[:, 0], pred_samples[:, 1], bins=bins, range=self.domain.cpu(), density=True)

        device = next(self.parameters()).device

        xx, yy = torch.meshgrid(torch.linspace(-1, 1, bins), torch.linspace(-1, 1, bins))
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.to(device)
        log_prob = target.log_prob(zz).view(*xx.shape)
        prob = torch.exp(log_prob)

        return (torch.from_numpy(H).to(device) - prob).square().mean().sqrt()

    
###########################################################################################
### Naive Flow Model
###########################################################################################

class  NaiveForwardModel(AbstractModel):
    def __init__(self, features, prior, encoding, layers, domain = [[0, 1], [0, 1]]):
        super().__init__()
        self.prior = prior
        self.encoding = encoding
        self.layers = layers
        encoding_size = encoding.n_output_dims if type(encoding) == tcnn.Encoding else encoding.encoding_size
        self.network = MLP(encoding_size, features, layers, nn.SiLU())
        self.jac_func = torch.func.vmap(torch.func.jacrev(self, chunk_size=1))

        domain = torch.as_tensor(domain)

        self.register_buffer("domain", domain)

        self.alpha = 1e-3

    def forward(self, x: torch.Tensor):
        unsqueezed = False
        if len(x.shape) == 1:
            unsqueezed = True
            x = x.unsqueeze(0)

        # renormalize domain into [0, 1], everything outside will hash index into some table position, but it should be unlikely
        x = (x - self.domain[:, 0]) / (self.domain[:, 1] - self.domain[:, 0])

        enc = self.encoding(x)
        res = x + self.network(enc)
        
        res = self.domain[:, 0] + res * (self.domain[:, 1] - self.domain[:, 0])
        # res = wrap(res, self.domain[:, 1])

        if unsqueezed:
            return res.squeeze(0)
        return res
    
    def sample(self, num_samples: int):
        z = self.prior.sample(num_samples)
        return self(z)
    
    def compute_batch_gradient(self, target, batch, prior_pdf, batch_pdf):   
        jac_func = torch.func.vmap(torch.func.jacrev(self, chunk_size=1))    
         
        ### Compute jacobians of transform
        jacobians = jac_func(batch)
        # abs_dets = torch.det(jacobians).clamp(min = 1e-4)
        abs_dets = torch.det(jacobians).abs() + 1e-4

        ### Compute f(T)
        y = self(batch)
        probs_f = target.log_prob(y).exp()


        domain_loss_factor = 0
        domain_loss = torch.maximum(torch.maximum(self.domain[:, 0] - y, y - self.domain[:, 1]), torch.zeros_like(y)).sum(-1)

        ### Compute loss
        losses = -probs_f * torch.log(prior_pdf / abs_dets) * prior_pdf / batch_pdf + domain_loss_factor * domain_loss
        loss = torch.mean(losses[~(torch.isnan(losses) | torch.isinf(losses))])
        
        ### Backpropagation
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()

        return loss

    def visualize(self, target, grid_size = 100, num_samples = 10000, colormap = 'jet', samples_in_domain = True):
        xx, yy = torch.meshgrid(
            torch.linspace(self.domain[0][0], self.domain[0][1], grid_size), 
            torch.linspace(self.domain[1][0], self.domain[1][1], grid_size)
        )
        
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.to(next(self.parameters()).device)

        log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
        prob = torch.exp(log_prob)


        q=self.prior.log_prob(zz).exp().cpu().view(*xx.shape)
        

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        ax1.pcolormesh(xx, yy, prob.cpu().detach(), cmap=colormap)
        ax1.set_title("target")
        
        ax2.pcolormesh(xx, yy, q, cmap=colormap)
        ax2.set_title("prior")
                
        x = self.prior.sample(num_samples)
        pred_samples = self(x).cpu().detach().numpy()
        if samples_in_domain:
            ax3.hist2d(pred_samples[:, 0], pred_samples[:, 1], bins=100, range=self.domain.cpu(), cmap=colormap, norm="linear")
        else:
            ax3.hist2d(pred_samples[:, 0], pred_samples[:, 1], bins=100, cmap=colormap, norm="linear")
        ax3.set_title("drawn samples")

        return fig
    
    def RMSE(self, target, num_samples = 1000000, bins = 50):
        x = self.prior.sample(num_samples)
        pred_samples = self(x).cpu().detach().numpy()

        H, _, _ = np.histogram2d(pred_samples[:, 0], pred_samples[:, 1], bins=bins, range=self.domain.cpu(), density=True)

        device = next(self.parameters()).device

        xx, yy = torch.meshgrid(torch.linspace(-1, 1, bins), torch.linspace(-1, 1, bins))
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.to(device)
        log_prob = target.log_prob(zz).view(*xx.shape)
        prob = torch.exp(log_prob)

        return (torch.from_numpy(H).to(device) - prob).square().mean().sqrt()
