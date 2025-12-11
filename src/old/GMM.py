import torch
from torch import nn


class GMM(nn.Module):
    def __init__(self, num_gaussians, 
                 dims = 2, 
                 dtype = torch.float32, 
                 init_means = None, 
                 init_cholesky = None, # Important: Diagonal entries will be epxonentiated to be valid cholesky decomposition
                 init_weight_logits = None):
        super().__init__()
        
        # number of individual gaussians
        self.num_gaussians = num_gaussians

        # dimension of the multivariate gaussian
        self.D = dims

        # means of the gaussians
        if init_means is None:
            self.means = nn.Parameter(torch.rand(self.num_gaussians, self.D, dtype = dtype))
        else:
            self.means = nn.Parameter(init_means.type(dtype))

        # indices into the lower triangular of a DxD Matrix
        tril_indices = torch.tril_indices(self.D, self.D, offset = 0)

        # create parameters for cholesky decomposition to learn covariances
        if init_cholesky is None:
            self.cholesky_params = nn.Parameter(
                torch.randn(self.num_gaussians, len(tril_indices[0]))
            )
        else:
            self.cholesky_params = nn.Parameter(init_cholesky.type(dtype))


        # weights of the mixtures. Softmax will be applied to these, so zeroes will be mapped to 1 / self.num_gaussians at init
        if init_weight_logits is None:
            self.weight_logits = nn.Parameter(torch.zeros(self.num_gaussians, dtype = dtype))
        else:
            self.weight_logits = nn.Parameter(init_weight_logits.type(dtype))

        # register the lower triangular element indices as buffers, we will read from them later
        self.register_buffer("tril_row_idx", tril_indices[0])
        self.register_buffer("tril_col_idx", tril_indices[1])

        # constant term for computing the log determinant of the multivariate normal
        const_term = -0.5 * self.D * torch.log(torch.tensor(2 * torch.pi))
        self.register_buffer('const_term', const_term)

    """ Returns the matrices L from the cholesky decomposition, where the covariance matrix A = L*L^T"""
    def get_L_matrix(self):
        # Create DxD matrices and write the cholesky factors at the positions specified by the tril_indices
        matrices = torch.zeros(self.num_gaussians, self.D, self.D, device = self.means.device, dtype = self.means.dtype)
        matrices[: , self.tril_row_idx, self.tril_col_idx] = self.cholesky_params

        # Need to ensure that diagonal entries of cholesky decomposition are positive
        diag_idx = torch.arange(self.D, device = self.means.device)
        matrices[:, diag_idx, diag_idx] = torch.exp(matrices[:, diag_idx, diag_idx])
        return matrices

    def sample(self, num_samples):
        # sample mixture IDs
        # Docs say to use log_softmax ? only for NLLLoss ?
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        mixture_idx = torch.multinomial(torch.nn.functional.softmax(self.weight_logits, dim = 0), 
                                        num_samples=num_samples, 
                                        replacement=True)

        # Get the means and covariances for the sampled gaussians 
        batch_means = self.means[mixture_idx]
        batch_L = self.get_L_matrix()[mixture_idx]

        # Sample from a standard normal
        z = torch.randn(num_samples, self.D, device=self.means.device, dtype = self.means.dtype)

        # transform standard normal samples to target gaussians
        # unsqueeze and squeeze to because bmm expects matching dimensions
        return batch_means + torch.bmm(batch_L, z.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, x):
        # get weights of the mixtures [K]
        weights = torch.nn.functional.softmax(self.weight_logits, dim = 0)

        ### where K = self.num_gaussians
        # x has dim: [n, D] -> [n, 1, D]
        # means has dim: [K, D] -> [1, K, D]
        # subtraction is then broadcast
    
        # for each sample (first dimension), and each gaussian (second dimension), difference in each feature (third dimension)
        # final dim: [n, K, D]
        diff = x.unsqueeze(1) - self.means.unsqueeze(0) ## match dimensions

        # get all covariance matrices [K, D, D]
        cov_matrices = self.get_L_matrix()

        expanded_mats = cov_matrices.unsqueeze(0)
        expanded_diff = diff.unsqueeze(-1)

        ### 
        # broadcast over sample count (dim = 0) and number of gaussians (dim = 1)
        # Also add singleton dimension at end of diff to transpose vector
        # Remove this dimension by squeezing after 
        ys = torch.linalg.solve_triangular(expanded_mats, 
                                           expanded_diff, 
                                           upper = False).squeeze(-1)

        # get result ys in [n, K, D]
        # Sum over squares along D (now dim = 1) [m, K]
        mahal_sq = (ys ** 2).sum(dim = 2)

        # log of the determinant of the covariance matrices, [K]
        log_det = torch.log(torch.diagonal(cov_matrices, dim1=-2, dim2=-1)).sum(dim = 1)

        # log probabilities for each gaussian [n, K]
        log_prob = self.const_term - log_det.unsqueeze(0) - 0.5 * mahal_sq

        # weigh each prob by the weight of its gaussian; becomes a sum in log space [n, K]
        weighted_log_prob = log_prob + torch.log(weights.unsqueeze(0))

        # sum over weighted probabilities: needs to be done in linear space -> use logsumexp [n]
        return torch.logsumexp(weighted_log_prob, dim = 1)

    def prob(self, x):
        return torch.exp(self.log_prob(x))
    
    """Adding the L2 regularization to the loss during early training can help prevent mode collapse"""
    def weight_L2(self):
        # TODO test out if normalized or unnormalized weights work better ?

        return torch.nn.functional.softmax(self.weight_logits, dim = 0).square().sum()
        # return self.weight_logits.exp().square().sum()