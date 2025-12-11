import numpy as np
import torch

one_filter = np.vectorize(lambda x: 0.0 if abs(x - 1) < 1e-4 else x)

def MCMC_large_step():
    return np.random.rand(2)

def MCMC_small_step(curr_sample, step_size):
    # Small random step to go to with MC Sampling
    rand_vec = (np.random.rand(*curr_sample.shape) * 2 - 1) * step_size
    return curr_sample + rand_vec

def MCMC_step(dist, curr_sample, alpha, step_size):
    if np.random.rand(1)[0] < alpha:
        tent_sample = MCMC_large_step()
    else:
        tent_sample = MCMC_small_step(curr_sample, step_size)
    
    tent_sample %= 1
    tent_sample = one_filter(tent_sample)

    curr_eval = dist.eval(curr_sample)
    if curr_eval < 1e-4:
        return tent_sample

    a = min(1, dist.eval(tent_sample) / curr_eval)
    if np.random.rand(1)[0] < a:
        return tent_sample
    else:
        return curr_sample
    
def MCMC_large_step_torch():
    return torch.rand(2)

def MCMC_small_step_torch(curr_sample, step_size):
    # Small random step to go to with MC Sampling
    rand_vec = (torch.rand(*curr_sample.shape) * 2 - 1) * step_size
    return curr_sample + rand_vec

def MCMC_step_torch(dist, curr_sample, alpha, step_size):
    if np.random.rand(1)[0] < alpha:
        tent_sample = MCMC_large_step_torch()
    else:
        tent_sample = MCMC_small_step_torch(curr_sample, step_size)
    
    tent_sample %= 1
    for d in tent_sample:
        if d == 1:
            d = 0

    curr_eval = dist.eval(curr_sample)
    if curr_eval < 1e-4:
        return tent_sample

    a = min(1, dist.eval(tent_sample) / curr_eval)
    if np.random.rand(1)[0] < a:
        return tent_sample
    else:
        return curr_sample