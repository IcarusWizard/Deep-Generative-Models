import torch
from torch.functional import F
import numpy as np

from .modules import Dequantization, GlowAffineStep, GlowAdditiveStep
from .utils import LOG2PI, squeeze, unsqueeze 
from .trainer import FlowTrainer

class GLOW(torch.nn.Module):
    def __init__(self, c=3, h=32, w=32, features=256, K = 16, L = 3, bits=8, constraint=0.9, coupling='affine', use_lu=True):
        super(GLOW, self).__init__()
        self.in_channels = c
        self.in_h = h
        self.in_w = w
        self.bits = bits
        self.L = L
        self.z_size = h * w * c

        self.prior = torch.distributions.Normal(0, 1) 

        self.pre_process = Dequantization(constraint=constraint, bits=bits)

        if coupling == 'affine':
            GlowStep = GlowAffineStep
        elif coupling == 'additive':
            GlowStep = GlowAdditiveStep
        else:
            raise ValueError('coupling {} is not supported'.format(coupling))

        blocks = []
        for i in range(L):
            h = h // 2
            w = w // 2
            c = c * 4
            blocks.append(torch.nn.ModuleList([GlowStep(c, features, use_lu) for j in range(K)]))
            c = c // 2

        self.blocks = torch.nn.ModuleList(blocks)

        self.out_channels = c * 2
        self.out_h = h
        self.out_w = w

    def forward(self, x, reverse=False, average=True):
        if reverse:
            z = x
            x = None
            z_chunks = []
            for i in range(self.L - 1):
                chunk, z = torch.chunk(z, 2, dim=1)
                z_chunks.append(chunk)

            for block in reversed(self.blocks):
                # revserse split
                if x is None:
                    x = z.view(z.shape[0], self.out_channels, self.out_h, self.out_w)
                else:
                    z = z_chunks.pop()
                    z = z.view(*x.shape)
                    x = torch.cat([x, z], dim=1)

                # reverse steps
                for step in reversed(block):
                    x = step.backward(x)

                # reverse squeeze
                x = unsqueeze(x)

            return self.pre_process.backward(x)        
        else:
            z = None
            h, LL = self.pre_process(x)

            for i, block in enumerate(self.blocks):
                # Squeeze
                h = squeeze(h)
                
                # Steps of flow
                for step in block:
                    h, ll = step(h)
                    LL = LL + ll

                # Split if it is not the last layer
                if i < self.L - 1:
                    h, _z = torch.chunk(h, 2, dim=1)
                    _z = _z.view(_z.shape[0], -1)
                    if z is None:
                        z = _z
                    else:
                        z = torch.cat([z, _z], dim=1)
                else:
                    h = h.view(h.shape[0], -1)
                    z = torch.cat([z, h], dim=1)

            LL = LL + torch.sum(self.prior.log_prob(z), dim=1)

            NLL = - LL + self.bits * np.prod(self.z_size) * np.log(2)

            if average:
                NLL = torch.mean(NLL)

            return z, NLL

    def x2z(self, x):
        z, _ = self.forward(x)

        return z

    def z2x(self, z):
        return self.forward(z, reverse=True)

    def sample(self, num_samples=1000, temperature=1.0):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        z = torch.randn(num_samples, self.z_size, device=device, dtype=dtype) * temperature

        return self.z2x(z)

    def get_trainer(self):
        return FlowTrainer