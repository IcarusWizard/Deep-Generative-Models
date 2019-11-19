import numpy as np
import torch, torchvision, math
from torch.functional import F
from torch.nn import Parameter, init
import matplotlib.pyplot as plt
import pickle, torch, math, random, os
import PIL.Image as Image

from ..utils import LOG2PI

def squeeze(x):
    """
        Squeeze tensor x[B, C, S, S] to y[B, C*4, S//2, S//2]
    """
    # this works like a dark magic
    [B, C, H, W] = list(x.size())
    x = x.reshape(B, C, H//2, 2, W//2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, C*4, H//2, W//2)
    return x

def unsqueeze(y):
    """
        Unsqueeze tensor y[B, C*4, S//2, S//2] to x[B, C, S, S]
    """
    [B, C, H, W] = list(y.size())
    y = y.reshape(B, C//4, 2, 2, H, W)
    y = y.permute(0, 1, 4, 2, 5, 3)
    y = y.reshape(B, C//4, H*2, W*2)
    return y

def build_chessboard_mask(h, w):
    h_axis = np.repeat(np.arange(h), w).reshape(w, h).T
    w_axis = np.repeat(np.arange(w), h).reshape(h, w)

    mask = (h_axis % 2 + w_axis) % 2

    return mask.astype(np.float32)

def build_channel_mask(c):
    mask = np.ones(c)
    mask[:c//2] = 0

    return mask.astype(np.float32)