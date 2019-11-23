import numpy as np
import torch, torchvision, math
from torch.functional import F
from torch.nn import Parameter, init
import matplotlib.pyplot as plt
import pickle, torch, math, random, os
import PIL.Image as Image

class MaxOut(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear = torch.nn.Linear(in_channel, 2 * out_channel)

    def forward(self, x):
        out = self.linear(x)
        return torch.max(*torch.chunk(x, 2, dim=1))
