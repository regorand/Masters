from xml.etree.ElementInclude import include
import torch
import tinycudann as tcnn

"""
Implementation of https://arxiv.org/abs/1906.04032
It is limited to 2 dimensions though, as that simplifies the implementation and is enough 
for the purposes in this project

This implementation is in part based on https://github.com/VincentStimper/normalizing-flows

"""

# @torch.jit.script
def build_spline_params(input, B: float, beta: float = 1.0, include_phase: bool = False):
    # if (input.shape[-1] + 1) % 3 != 0: ### Old version
    if (input.shape[-1] - 1) % 3 != 0:
        # raise RuntimeError("Spline needs input of size 3K - 1 in last dim")
        raise RuntimeError("Spline needs input of size 3K + 1 in last dim")

    # K = int((input.shape[-1] + 1) / 3) ### Old version
    K = int((input.shape[-1] - 1) / 3)

    theta_w = input[..., :K]
    theta_h = input[..., K:2 * K]
    theta_d = input[..., 2 * K:] if not include_phase else input[..., 2 * K:3 * K]

    x_left = -B
    y_left = -B

    widths =  torch.nn.functional.softmax(theta_w, dim = -1) * 2 * B
    heights = torch.nn.functional.softmax(theta_h, dim = -1) * 2 * B

    ### Old version, can always come back to it

    # knot_derivs = torch.cat([
    #     torch.ones_like(theta_d, dtype=torch.float32),
    #     torch.nn.functional.softplus(theta_d, beta=beta),
    #     torch.ones_like(theta_d, dtype=torch.float32)
    # ], dim=-1)

    ### End of old version


    ### Fix to old version

    # knot_derivs = torch.cat([
    #     torch.ones((theta_d.shape[0], 1), dtype=torch.float32),
    #     torch.nn.functional.softplus(theta_d, beta=beta),
    #     torch.ones((theta_d.shape[0], 1), dtype=torch.float32)
    # ], dim=-1)

    ### End of fix to old version


    ### New version

    knot_derivs = torch.nn.functional.softplus(theta_d, beta=beta)

    if include_phase:
        knot_derivs = torch.cat([knot_derivs, knot_derivs[..., :1]], dim=-1)

    phase = torch.full(input.shape[:-2], 0.0) if not include_phase else input[..., -1:]

    zeros_shape = list(widths.shape[:-1]) + [1]

    vertices_x = torch.cat([
            torch.zeros(zeros_shape, dtype=widths.dtype, device=widths.device),
            torch.cumsum(widths, dim=-1)
        ], dim=-1) + x_left


    vertices_y = torch.cat([
            torch.zeros(zeros_shape, dtype=heights.dtype, device=heights.device),
            torch.cumsum(heights, dim=-1)
        ], dim=-1) + y_left
    
    return vertices_x, vertices_y, knot_derivs, phase

def mod_to_domain(x, B):
    return torch.fmod(torch.fmod(x + B, B * 2) + B * 2, B * 2) - B
    

# @torch.jit.script
def eval_quadratic_spline(vertices_x, vertices_y, knot_derivs, x, B, phase):
    idx = torch.searchsorted(vertices_x, x) - 1
    idx = torch.clamp(idx, 0, vertices_x.shape[-1] - 2)

    mask = (x >= vertices_x[..., [0]]) & (x <= vertices_x[..., [-1]])

    y_i = torch.gather(vertices_y, dim=-1, index=idx)
    x_i = torch.gather(vertices_x, dim=-1, index=idx)

    y_diff = torch.gather(vertices_y, dim=-1, index=idx + 1) - y_i
    x_diff = torch.gather(vertices_x, dim=-1, index=idx + 1) - x_i

    d_i = torch.gather(knot_derivs, dim=-1, index=idx)
    d_iplus1 = torch.gather(knot_derivs, dim=-1, index=idx + 1)

    sk = y_diff / x_diff
    xeta = (x - x_i) / x_diff

    numerator   = y_diff * (sk * xeta ** 2 + d_i * xeta * (1 - xeta))
    denominator = sk + (d_iplus1 + d_i - 2 * sk) * xeta * (1 - xeta)

    y = y_i + numerator / denominator
    y = torch.where(mask, y, x)

    deriv_num = sk**2 * (d_iplus1 * xeta**2 + 2 * sk * xeta * (1 - xeta) + d_i * (1 - xeta)**2)

    log_det = torch.where(mask, torch.log(deriv_num) - torch.log(denominator) * 2, 0)
    # log_det = torch.log(deriv_num) - torch.log(denominator) * 2

    return mod_to_domain(y + phase, B), log_det.squeeze(-1)

# @torch.jit.script
def invert_quadratic_spline(vertices_x, vertices_y, knot_derivs, y, B, phase):
    idx = torch.searchsorted(vertices_y, y) - 1
    idx = torch.clamp(idx, 0, vertices_y.shape[-1] - 2)

    mask = (y >= vertices_y[..., [0]]) & (y <= vertices_y[..., [-1]])


    y_i = torch.gather(vertices_y, dim=-1, index=idx)
    x_i = torch.gather(vertices_x, dim=-1, index=idx)

    y_diff = torch.gather(vertices_y, dim=-1, index=idx + 1) - y_i
    x_diff = torch.gather(vertices_x, dim=-1, index=idx + 1) - x_i

    d_i = torch.gather(knot_derivs, dim=-1, index=idx)
    d_iplus1 = torch.gather(knot_derivs, dim=-1, index=idx + 1)

    sk = y_diff / x_diff

    sample_diff = y - y_i
    derivs_min_2s = (d_iplus1 + d_i - 2 * sk)


    a = y_diff * (sk - d_i) + sample_diff * derivs_min_2s
    b = y_diff * d_i - sample_diff * derivs_min_2s
    c = -sk * sample_diff

    xeta = (2 * c / (-b - torch.sqrt(b ** 2 - 4 * a * c)))

    x = xeta * x_diff + x_i
    x = torch.where(mask, x, y)

    denominator = sk + (d_iplus1 + d_i - 2 * sk) * xeta * (1 - xeta)
    deriv_num = sk**2 * (d_iplus1 * xeta**2 + 2 * sk * xeta * (1 - xeta) + d_i * (1 - xeta)**2)

    log_det = torch.where(mask, torch.log(deriv_num) - torch.log(denominator) * 2, 0)
    # log_det = torch.log(deriv_num) - torch.log(denominator) * 2

    return mod_to_domain(x - phase, B), -log_det.squeeze(-1)


class NeuralSplineFlow(torch.nn.Module):
    def __init__(self, K, context_size, enc_config, net_config, latent_to_hash_grid, beta = 1.0, with_phase: bool = False):
        super().__init__()
        self.B = 1.0 # hardcode to range [-1, 1], scaling can be done outside componentwise. Otherwise it's more complicated for non-square domains
        self.K = K
        self.beta = beta # Beta for softplus function to normalize the vertex derivatives

        self.context_size = context_size

        ### Seperated
        self.encoding = tcnn.Encoding(context_size + 1, enc_config)
        self.network = tcnn.Network(self.encoding.n_output_dims, (3 * K + 1), net_config)

        ### Combined
        # self.network = tcnn.NetworkWithInputEncoding(context_size + 1, (3 * K + 1), enc_config, net_config)

        self.network.jit_fusion = tcnn.supports_jit_fusion()
        self.latent_to_hash_grid = latent_to_hash_grid
        self.with_phase = with_phase


    def to_encoding_domain(self, x):
        """
        The quadratic splines are defined on the range [-B, B], but the input encodings to the neural network
        expect its inputs to be in the range [0, 1]. 
        """
        return (x + self.B) / (2 * self.B)

    def forward(self, x, context = None):
        x0 = x[:,  :1]
        x1 = x[:, 1: ]

        if context is None:
            context = torch.empty((0), device = x.device)

        if self.latent_to_hash_grid:
            net_inp = (torch.concat((context.expand(x0.shape[0], context.shape[1]), self.to_encoding_domain(x0)), dim = -1)
                    if context.shape[0] > 0
                    else (self.to_encoding_domain(x0)))
        else:
            net_inp = (torch.concat((self.to_encoding_domain(x0), context.expand(x0.shape[0], context.shape[1])), dim = -1)
                    if context.shape[0] > 0
                    else (self.to_encoding_domain(x0)))

        ### Seperated
        params = self.network(self.encoding(net_inp)).squeeze(-1)
        
        ### Combined
        # params = self.network(net_inp).squeeze(-1)

        vertices_x, vertices_y, knot_derivs, phase = build_spline_params(params, self.B, self.beta, include_phase=self.with_phase)

        y1, abs_log_det = eval_quadratic_spline(vertices_x, vertices_y, knot_derivs, x1, self.B, phase)

        y = torch.cat((
            x0,
            y1
        ), dim = -1)

        return y, abs_log_det
    
    def inverse(self, y, context = None):
        y0 = y[:,  :1]
        y1 = y[:, 1: ]

        if context is None:
            context = torch.empty((0), device = y.device)

        if self.latent_to_hash_grid:
            net_inp = (torch.concat((context.expand(y0.shape[0], context.shape[1]), self.to_encoding_domain(y0)), dim = -1)
                        if context.shape[0] > 0
                        else (self.to_encoding_domain(y0)))
        else:
            net_inp = (torch.concat((self.to_encoding_domain(y0), context.expand(y0.shape[0], context.shape[1])), dim = -1)
                        if context.shape[0] > 0
                        else (self.to_encoding_domain(y0)))

        ### Seperated
        params = self.network(self.encoding(net_inp)).squeeze(-1)
        
        ### Combined
        # params = self.network(net_inp).squeeze(-1)

        vertices_x, vertices_y, knot_derivs, phase = build_spline_params(params, self.B, self.beta, self.with_phase)

        x1, abs_log_det = invert_quadratic_spline(vertices_x, vertices_y, knot_derivs, y1, self.B, phase)

        x = torch.cat((
            y0,
            x1
        ), dim = -1)

        return x, abs_log_det    
    



def pretrain_flow(flow: NeuralSplineFlow, iterations = 100, batch_size = 64):
    context_size = flow.context_size

    print(f"pretraining flow with {iterations} its!")

    width_target = 0
    height_target = 0
    d_target = 1
    # d_target = log(exp(flow.beta) - 1) / flow.beta

    optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-2, weight_decay=1e-4)

    for it in range(iterations):
        input = torch.rand((batch_size, 1), device = 'cuda')

        context = torch.zeros((batch_size, context_size), device = 'cuda')
        # context = torch.randn((batch_size, context_size), device = 'cuda')

        net_inp = torch.concat((input, context), dim = -1)

        params = flow.network(net_inp).squeeze(-1)

        K = int((params.shape[-1] + 1) / 3)

        theta_w = params[..., :K]
        theta_h = params[..., K:2 * K]
        theta_d = params[..., 2*K:]

        derivs = torch.nn.functional.softplus(theta_d, beta=flow.beta)

        loss_w = ((theta_w - width_target) ** 2).mean().sqrt()
        loss_h = ((theta_h - height_target) ** 2).mean().sqrt()
        loss_d = ((derivs - d_target) ** 2).mean().sqrt()

        loss = loss_w + loss_h + loss_d

        loss.backward()
        optimizer.step()

       


class NormalizingFlow(torch.nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = torch.nn.ModuleList(flows)

    def permute(self, x):
        x[..., [0, 1]] = x[..., [1, 0]]
        return x
    
    def forward(self, z, context = None):
        for flow in self.flows:
            z, _ = flow(z, context=context)
            z = self.permute(z)

        if len(self.flows) % 2 != 0:
            return self.permute(z)

        return z
    
    def forward_and_log_det(self, z, context = None):
        log_det = torch.zeros(len(z), device=z.device)
    
        for flow in self.flows:
            z, log_d = flow(z, context=context)
            z = self.permute(z)
            log_det += log_d

        if len(self.flows) % 2 != 0:
            return self.permute(z), log_det

        return z, log_det
    
    def inverse(self, x, context = None):
        for i in range(len(self.flows) - 1, -1, -1):
            x = self.permute(x)
            x, _ = self.flows[i].inverse(x, context=context)
            
        if len(self.flows) % 2 != 0:
            return self.permute(x)

        return x

    def inverse_and_log_det(self, x, context = None):
        log_det = torch.zeros(len(x), device=x.device)

        for i in range(len(self.flows) - 1, -1, -1):
            x = self.permute(x)
            
            x, log_d = self.flows[i].inverse(x, context=context)
            log_det += log_d
            
        if len(self.flows) % 2 != 0:
            return self.permute(x), log_det

        return x, log_det

    def log_prob(self, x, context = None):
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x

        for i in range(len(self.flows) - 1, -1, -1):
            z = self.permute(z)

            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
            

        return log_q