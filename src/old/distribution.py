import cv2
import torch
import numpy as np
import src.old.parameter_model as param_model
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections.abc import Callable
import src.old.utils as utils

def vis_dist_by_rejection(dist, N):
    # dist_points = sample_dist_rejection(2000)
    dist_points = dist.rejection_sample(N)

    dims = dist.get_dims()

    plt.figure(figsize=(dims[0] / 100, dims[1] / 100))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    plt.scatter(dist_points[:, 0], dist_points[:, 1], color="black", marker="o")
    ax = plt.gca()
    ax.set_xlim([0, 1])
    ax.set_ylim([1, 0])
    plt.show()


@dataclass
class InputDistribution:
    sample: Callable[[int], np.ndarray] = utils.sample_uniform_points
    transform: Callable[[np.ndarray], np.ndarray] = lambda x: x

Uniform_Input_Distribution = InputDistribution(utils.sample_uniform_points, lambda x: x)

def build_param_distribution(params: param_model.Param_Gauss_model) -> InputDistribution:
    return InputDistribution(lambda count : param_model.sample_param_model(params, count),
                             lambda points: param_model.unif_to_model(params, points))


class Distribution:
    def __init__(self, input_file):
        self.dist = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
        self.width = self.dist.shape[1]
        self.height = self.dist.shape[0]
        
    # Can flip here because cv2 loads with y dim in first index
    def get_dims(self, flip=False):
        if flip:
            return (self.height, self.width)    
        return (self.width, self.height)

    def eval(self, point):
        if point[1] < 0 or point[1] >= 1 or point[0] < 0 or point[0] >= 1:
            return 0
        return self.dist[int(point[1] * self.height)][int(point[0] * self.width)]
    
    def eval_np(self, points):
        if type(points) is torch.Tensor:
            points = points.numpy()
        point_indices = (np.flip(self.dist.shape) * points).astype(int)
        return self.dist[point_indices[:, 1], point_indices[:, 0]]

    def log_prob(self, point):
        val = self.eval(point)
        if val < 1e-8:
            return -1e8
        return np.log(val)

    def to_torch(self):
        self.torch_dist = torch.Tensor(self.dist)

    def has_torch(self):
        return hasattr(self, "torch_dist")
    
    def fit_gaussian(self):
        self.param_model = param_model.fit_gaussian(self)
        return self.param_model

    def get_gaussian(self):
        return self.param_model

    def has_gaussian_fit(self):
        return hasattr(self, "param_model")

    def eval_torch(self, points):
        if not self.has_torch():
            raise Exception("Distribution does not have a torch Tensor yet, call Distribution.to_torch()")

        return self.torch_dist[torch.clamp((self.height * points[:, 1]).int(), 0, self.height - 1), torch.clamp((self.width * points[:, 0]).int(), 0, self.width - 1)]

    def rejection_sample(self, N):
        points = []
        while len(points) < N:
            point = np.random.rand(2)
            point_eval = self.eval(point) / 255
            rand_0_1 = np.random.rand()
            if point_eval > rand_0_1:
                points.append(point)
        return np.array(points)
    
    def squared_dist(self, other, normalize = True, return_elementwise = False):
        if self.dist.shape != other.shape:
            # TODO could transform them to compare them -> needs functionality somewhere else though
            raise Exception("Can't compare distributions of different shapes")

        count = 1
        if normalize:
            for d in self.dist.shape:
                count *= d
        
        norm_dist = self.dist.astype(np.float64) / 255

        if return_elementwise:
            return np.square(np.subtract(norm_dist, other)) / count
        else:
            return np.sum(np.square(np.subtract(norm_dist, other)) / count)
        
    def abs_dist(self, other, normalize = True, return_elementwise = False):
        if self.dist.shape != other.shape:
            # TODO could transform them to compare them -> needs functionality somewhere else though
            raise Exception("Can't compare distributions of different shapes")

        count = 1
        if normalize:
            for d in self.dist.shape:
                count *= d

        norm_dist = self.dist.astype(np.float64) / 255        
        if return_elementwise:
            return np.abs(np.subtract(norm_dist, other)) / count
        else:
            return np.sum(np.abs(np.subtract(norm_dist, other)) / count)