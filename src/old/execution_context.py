from dataclasses import dataclass, field
import src.old.domain as dmn
import src.old.distribution as distribution
import torch

@dataclass
class ExecutionContext:
    target_dist: distribution.Distribution
    input_dist: distribution.InputDistribution = field(default_factory = lambda: distribution.Uniform_Input_Distribution)
    domain: dmn.Domain = field(default_factory = lambda: dmn.R2_Domain)
    device: torch.device = torch.device("cpu")
