import torch
from torch.functional import F

from .modules import RowLSTM, BiLSTM, MaskConv
from .utils import build_maskA, build_maskB

class PixelRNN(torch.nn.Module):
    def __init__(self, c, h, w, mode='row', features=16, layers=7, post_features=1024, bits=8):
        super().__init__()
        self.c = c
        self.h = h
        self.w = w
        self.bits = bits

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

        output_channels = c * (2 ** bits) if bits != 1 else c
        self.output_conv = MaskConv(post_features, output_channels, (1, 1), 
            build_maskB(post_features, output_channels, (1, 1), group=c))

    def forward(self, x):        
        # build target
        target = (x * (2 ** self.bits - 1)).long()

        # compute conditional distributions
        distribution = self.get_distribution(x)

        if self.bits == 1: # bmnist
            log_likelihood = - F.binary_cross_entropy(distribution, target.float())
        else:
            log_likelihood = 0
            for i in range(3):
                log_likelihood -= F.nll_loss(distribution[i], target[:, i])
            log_likelihood /= 3

        return - log_likelihood

    def sample(self, num_samples):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = torch.zeros(num_samples, self.c, self.h, self.w, device=device, dtype=dtype)

        for i in range(self.h):
            for j in range(self.w):
                for k in range(self.c):
                    distribution = self.get_distribution(x)
                    if self.bits == 1: # bmnist
                        sample = (torch.rand(num_samples, device=device, dtype=dtype) < distribution[:, k, i, j]).float()
                        x[:, k, i, j] = sample
                    else:
                        distribution = torch.exp(distribution[k])
                        sample = torch.multinomial(distribution[:, :, i, j], 1)
                        x[:, k, i, j] = sample.view(-1).float() / (2 ** self.bits - 1)

        return x

    def get_distribution(self, x):
        h = self.input_conv(x)

        for lstm in self.lstms:
            h = lstm(h)

        h = self.post_conv(h)

        output = self.output_conv(h)

        if self.bits == 1: # bmnist
            return torch.sigmoid(output)
        else:
            return [F.log_softmax(chunk, dim=1) for chunk in torch.chunk(output, 3, dim=1)]