import numpy as np
import torch
from torch.functional import F

from ..utils import LOG2PI

def get_kl(mu, logs): # Compute the KL divergence between diagonal gauss
    if len(mu.shape) == 2:
        return torch.sum(0.5 * (mu ** 2 + torch.exp(logs) ** 2 - 2 * logs - 1), dim=1)
    else:
        return torch.sum(0.5 * (mu ** 2 + torch.exp(logs) ** 2 - 2 * logs - 1), dim=(1, 2, 3))
