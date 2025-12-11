from dataclasses import dataclass
import src.old.distribution as dist
import torch
from collections.abc import Callable
import numpy as np

def sample_uniform_points(count: int):
    return np.random.rand(count, 2)

@dataclass
class Context:
    def __init__(self, 
                target_dist, 
                lr = 1e-4,
                training_steps = 1000,
                batch_size = 64,
                device = "cpu",
                track_score = False,
                track_every = 1000,
                animate_training = False,
                training_gif_name = "gifs/training_vis.gif",
                gif_length = 4 # in seconds
                ):
        self.target_dist = target_dist
        self.lr = lr
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.track_score = track_score
        self.track_every = track_every
        self.animate_training = animate_training,
        self.training_gif_name = training_gif_name
        self.gif_length = gif_length

    target_dist: dist.Distribution
    lr: float
    training_steps: int
    batch_size: int
    device: torch.device
    track_score: bool 
    track_every: int
    animate_training: bool
    training_gif_name: str
    gif_length: int



@dataclass
class TrainingContext:
    lr: float = 1e-4
    training_steps: int = 1_000
    batch_size: int = 64
    track_score: bool = False
    track_every: int = 200
    animate_training: bool = False
    training_gif_name: str = "gifs/out.gif"
    gif_length: int = 5
