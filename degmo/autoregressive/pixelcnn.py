import torch
from torch.functional import F

from .modules import MaskConv, MaskRes, GatePixelCNNBlock
from .utils import build_maskA, build_maskB

class PixelCNN(torch.nn.Module):
    def __init__(self, c, h, w, mode='res', features=64, layers=7, filter_size=3, post_features=1024, bits=8):
        super().__init__()
        self.c = c
        self.h = h
        self.w = w
        self.bits = bits
        self.mode = mode

        output_channels = c * (2 ** bits) if bits != 1 else c
        if self.mode == 'res':
            real_feature = 2 * features
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
                if self.bits == 1: # bmnist
                    distribution = self.get_distribution(x)
                    sample = (torch.rand(num_samples, device=device, dtype=dtype) < distribution[:, 0, i, j]).float()
                    x[:, 0, i, j] = sample
                else:
                    if self.mode == 'res':
                        for k in range(self.c):
                            distribution = self.get_distribution(x)
                            distribution = torch.exp(distribution[k])
                            sample = torch.multinomial(distribution[:, :, i, j], 1)
                            x[:, k, i, j] = sample.view(-1).float() / (2 ** self.bits - 1)
                    else:
                        for k in range(self.c):
                            _distribution = torch.exp(distribution[k])
                            sample = torch.multinomial(_distribution[:, :, i, j], 1)
                            x[:, k, i, j] = sample.view(-1).float() / (2 ** self.bits - 1)

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

        if self.bits == 1: # bmnist
            return torch.sigmoid(output)
        else:
            return [F.log_softmax(chunk, dim=1) for chunk in torch.chunk(output, 3, dim=1)]