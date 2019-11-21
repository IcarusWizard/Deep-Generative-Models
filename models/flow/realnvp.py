import torch
from torch.functional import F
import numpy as np

from ..utils import bits2nats
from .modules import Dequantization, MultiLayerBlock, ChessboardCoupleLayer
    
class RealNVP2D(torch.nn.Module):
    def __init__(self, c=3, h=32, w=32, features=32, hidden_blocks=4, down_sampling=3, bits=8):
        super(RealNVP2D, self).__init__()
        self.in_channels = c
        self.in_h = h
        self.in_w = w
        self.bits = bits
        self.down_sampling = down_sampling
        self.z_size = c * h * w

        self.prior = torch.distributions.Normal(0, 1) 

        self.pre_process = Dequantization(bits=bits)

        blocks = []
        for i in range(down_sampling):
            blocks.append(MultiLayerBlock(c, h, w, features, hidden_blocks))
            h = h // 2
            w = w // 2
            c = c * 2

        self.blocks = torch.nn.ModuleList(blocks)

        self.final_chessboard_list = torch.nn.ModuleList([
            ChessboardCoupleLayer(c, h, w, i % 2, features, hidden_blocks) for i in range(4)
        ])

        self.out_channels = c
        self.out_h = h
        self.out_w = w

    def forward(self, x, reverse=False, average=True):
        if reverse:
            z = x
            z_chunks = []
            for i in range(self.down_sampling):
                chunk, z = torch.chunk(z, 2, dim=1)
                z_chunks.append(chunk)
            x = z.view(z.shape[0], self.out_channels, self.out_h, self.out_w)

            for coupling in reversed(self.final_chessboard_list):
                x = coupling.backward(x)

            for i, block in enumerate(reversed(self.blocks)):
                x = block.backward(x, z_chunks[-(i+1)])      

            return self.pre_process.backward(x)
        else:
            final_z = []
            z, logdet = self.pre_process(x)

            for block in self.blocks:
                z, _z, _logdet = block(z)
                logdet += _logdet
                final_z.append(_z.view(_z.shape[0], -1))

            for coupling in self.final_chessboard_list:
                z, _logdet = coupling(z)
                logdet += _logdet

            final_z.append(z.view(z.shape[0], -1))
            final_z = torch.cat(final_z, dim=1)

            LL = logdet + torch.sum(self.prior.log_prob(final_z), dim=1)

            NLL = - LL + bits2nats(self.bits) * self.z_size

            if average:
                NLL = torch.mean(NLL)

            return final_z, NLL

    def x2z(self, x):
        z, _ = self.forward(x)

        return z

    def z2x(self, z):
        return self.forward(z, reverse=True)

    def sample(self, num_samples=1000, temperature=1.0):
        device = next(self.parameters()).device
        
        z = torch.randn(num_samples, self.z_size, device=device, dtype=torch.float32) * temperature

        return self.z2x(z)