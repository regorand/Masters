from itertools import product
from scipy import stats
import numpy as np
from scipy.optimize import minimize
import src.old.distribution as distr
from dataclasses import dataclass
import matplotlib.pyplot as plt
import src.old.utils as utils

"""
Parameters for a single modal 2-variate gaussian fit
mx, my are the means
a, b, c are the parameters for a lower cholesky decomposition
The covariance matrix is computed from these by scipy.stats.Covariance.from_cholesky, which will look like:

[a^2       b^2
 b^2 b^2 + c^2]

 and is guaranteed to be positive semi definite
"""
@dataclass
class Param_Gauss_model:
    mx: float = 0.5
    my: float = 0.5 
    a: float = 0.1
    b: float = 0
    c: float = 0.1

def fit_gaussian(D):
    # Crude approximation of integral of the image to compute a normalized pdf
    norm_dist = np.divide(D.dist, D.dist.sum() / (D.width * D.height))

    x_data = np.stack(list(product(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))))
    y_data = np.stack([norm_dist[int(point[1] * D.height)][int(point[0] * D.width)] for point in x_data])

    def mul_gaussian(params):
        means = params[0: 2]
        cov_mat = stats.Covariance.from_cholesky([[params[2], 0], [params[3], params[4]]])

        estimated_y = stats.multivariate_normal.pdf(x_data, mean = means, cov = cov_mat, allow_singular=True)

        mse = abs(estimated_y - y_data).mean()

        return mse

    initParams = [0.5, 0.5, 0.1, 0, 0.1]
    results = minimize(mul_gaussian, initParams, method='Nelder-Mead')
    
    return Param_Gauss_model(*results.x)

def mean_cov_from_params(p_model: Param_Gauss_model):
    return ([p_model.mx, p_model.my], stats.Covariance.from_cholesky([[p_model.a, p_model.b], [p_model.b, p_model.c]]))

def sample_param_model(p_model: Param_Gauss_model, count: int):
    mean, cov = mean_cov_from_params(p_model)

    return stats.multivariate_normal(mean = mean, cov=cov).rvs(size = count)

def plot_param_model(p_model: Param_Gauss_model, ref_dist):
    mean, cov = mean_cov_from_params(p_model)

    samples = sample_param_model(p_model, 500)

    plt.xlim(left = 0, right = 1)
    plt.ylim(bottom = 0, top = 1)

    plt.scatter(samples[:, 0], samples[:, 1], color="blue", marker="o")

    sampled = ref_dist.rejection_sample(len(samples))
    plt.scatter(sampled[:, 0], sampled[:, 1], color="black", marker="o")

def unif_to_model(p_model, unif_points):
    s_norm_points = utils.box_mueller_transform(unif_points=unif_points)
    mean, cov = mean_cov_from_params(p_model)

    return [np.linalg.cholesky(cov.covariance).dot(p) + mean for p in s_norm_points]

