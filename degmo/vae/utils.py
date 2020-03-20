import numpy as np
import torch
from torch.functional import F

from ..utils import LOG2PI

def get_kl(mu, logs):
    """
        Compute the KL divergence between diagonal gauss
    """
    return torch.sum(0.5 * (mu ** 2 + torch.exp(logs) ** 2 - 2 * logs - 1), dim=[i for i in range(1, len(mu.shape))])
