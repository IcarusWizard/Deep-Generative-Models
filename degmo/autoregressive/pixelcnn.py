import torch
from torch.functional import F
import numpy as np

from .modules import MaskConv, MaskRes, GatePixelCNNBlock
from .utils import build_maskA, build_maskB

class PixelCNN(torch.nn.Module):
    """
        PixelCNN: 

            http://arxiv.org/abs/1601.06759
            http://arxiv.org/abs/1606.05328

        Inputs:

            c : int, channel of the input image
            h : int, height of the input image
            w : int, width of the input image
            mode : str, mode for convolution structure, choose from res and gate, default: res
            features : int, number of features in the convolutions, (h in the paper)
            layers : int, number of stacked convolution blocks
            filter_size : int, filter size used in convolution layers
            post_features : number of features in the convolutions after main blocks
            bits : int, information contained in each dimension, 2 ^ bits is equal to the categorical number
    """
    def __init__(self, c, h, w, mode='res', features=64, layers=7, filter_size=3, post_features=1024, bits=8):
        super().__init__()
        self.c = c
        self.h = h
        self.w = w
        self.bits = bits
        self.k = int(np.round(2 ** bits))
        self.mode = mode

        output_channels = c * self.k
        if self.mode == 'res':
            self.input_conv = MaskConv(c, c * 2 * features, (7, 7), 
                build_maskA(c, c * 2 * features, (7, 7), group=c), padding=(3, 3))

            self.blocks = torch.nn.ModuleList([MaskRes(h, w, c * features, self.c) for i in range(layers)])

            self.post_conv = torch.nn.Sequential(
                MaskConv(c * 2 * features, post_features, (1, 1), 
                    build_maskB(c * 2 * features, post_features, (1, 1), group=c)),
                torch.nn.ReLU(True),
                MaskConv(post_features, post_features, (1, 1), 
                    build_maskB(post_features, post_features, (1, 1), group=c)),
                torch.nn.ReLU(True),    
                MaskConv(post_features, output_channels, (1, 1), 
                    build_maskB(post_features, output_channels, (1, 1), group=c))          
            )
        else:
            self.blocks = torch.nn.ModuleList([GatePixelCNNBlock(c if i == 0 else features, features, filter_size) for i in range(layers)])

            self.output_conv = torch.nn.Sequential(
                torch.nn.Conv2d(features, post_features, 1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(post_features, post_features, 1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(post_features, output_channels, 1)
            )

    def forward(self, x):        
        # build target
        target = (x * (self.k - 1)).long()

        # compute conditional distributions
        distribution = self.get_distribution(x)

        log_likelihood = 0
        for i in range(self.c):
            log_likelihood -= F.nll_loss(distribution[i], target[:, i])
        log_likelihood /= self.c

        return - log_likelihood

    def sample(self, num_samples):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = torch.zeros(num_samples, self.c, self.h, self.w, device=device, dtype=dtype)

        for i in range(self.h):
            for j in range(self.w):
                if self.mode == 'res':
                    for k in range(self.c):
                        distribution = self.get_distribution(x)
                        distribution = torch.exp(distribution[k])
                        sample = torch.multinomial(distribution[:, :, i, j], 1)
                        x[:, k, i, j] = sample.view(-1).float() / (self.k - 1)
                else:
                    distribution = self.get_distribution(x)
                    for k in range(self.c):
                        _distribution = torch.exp(distribution[k])
                        sample = torch.multinomial(_distribution[:, :, i, j], 1)
                        x[:, k, i, j] = sample.view(-1).float() / (self.k - 1)

        return x

    def get_distribution(self, x):
        if self.mode == 'res':
            h = self.input_conv(x)

            for block in self.blocks:
                h = block(h) 
            
            output = self.post_conv(h)
        else:
            vertical, horizontal = x, x

            for block in self.blocks:
                horizontal, vertical = block(horizontal, vertical)

            output = self.output_conv(horizontal)

        return [F.log_softmax(chunk, dim=1) for chunk in torch.chunk(output, self.c, dim=1)]