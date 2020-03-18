import torch
from torch.functional import F
import numpy as np

from .modules import RowLSTM, BiLSTM, MaskConv
from .utils import build_maskA, build_maskB
from .trainer import AutoregressiveTrainer

class PixelRNN(torch.nn.Module):
    """
        PixelRNN: 

            http://arxiv.org/abs/1601.06759

        Inputs:

            c : int, channel of the input image
            h : int, height of the input image
            w : int, width of the input image
            mode : str, mode for convolution structure, choose from row and bi, default: row (NOTE: bi mode is quite slow)
            first_kernel_size : int, kernel size of the first convolution, default : 7 (same as the paper)
            features : int, number of features in the rnn, (h in the paper)
            layers : int, number of stacked rnn blocks
            post_features : number of features in the convolutions after main blocks
            bits : int, information contained in each dimension, 2 ^ bits is equal to the categorical number
    """
    def __init__(self, c, h, w, mode='row', first_kernel_size=7, features=16, layers=7, post_features=1024, bits=8):
        super().__init__()
        self.c = c
        self.h = h
        self.w = w
        self.bits = bits
        self.k = int(np.round(2 ** bits))
        self.mode = mode

        output_channels = c * self.k

        self.input_conv = MaskConv(c, c * 2 * features, (7, 7), 
            build_maskA(c, c * 2 * features, (7, 7), group=c), padding=(3, 3))

        if mode == 'row':
            self.lstms = torch.nn.ModuleList([RowLSTM(features * c, h, w, group=c) for _ in range(layers)])
        elif mode == 'bi':
            self.lstms = torch.nn.ModuleList([BiLSTM(features * c, h, w, group=c) for _ in range(layers)])
        else:
            raise ValueError('{} mode is not supported!'.format(mode))

        self.post_conv = torch.nn.Sequential(
            MaskConv(c * 2 * features, post_features, (1, 1), 
                build_maskB(c * 2 * features, post_features, (1, 1), group=c)),
            torch.nn.ReLU(True),
            MaskConv(post_features, post_features, (1, 1), 
                build_maskB(post_features, post_features, (1, 1), group=c)),
            torch.nn.ReLU(True),              
        )

        self.output_conv = MaskConv(post_features, output_channels, (1, 1), 
            build_maskB(post_features, output_channels, (1, 1), group=c))

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
                for k in range(self.c):
                    distribution = self.get_distribution(x)
                    distribution = torch.exp(distribution[k])
                    sample = torch.multinomial(distribution[:, :, i, j], 1)
                    x[:, k, i, j] = sample.view(-1).float() / (self.k - 1)

        return x

    def get_distribution(self, x):
        h = self.input_conv(x)

        for lstm in self.lstms:
            h = lstm(h)

        h = self.post_conv(h)

        output = self.output_conv(h)

        return [F.log_softmax(chunk, dim=1) for chunk in torch.chunk(output, self.c, dim=1)]

    def get_trainer(self):
        return AutoregressiveTrainer