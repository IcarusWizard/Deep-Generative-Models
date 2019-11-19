import torch
from torch.functional import F
import numpy as np

from .modules import CoupleBlock1D, CoupleSigmoid, LOG2PI, CoupleLayers1D, ChessboardCoupleBlock, ChannelCoupleBlock, Dequantization
from .utils import squeeze, unsqueeze
    
class RealNVP2D(torch.nn.Module):
    def __init__(self, c=3, h=32, w=32, features=256, hidden_blocks=8, bits=8):
        super(RealNVP2D, self).__init__()
        self.in_channels = c
        self.in_h = h
        self.in_w = w
        self.bits = bits
        self.z_size = c * h * w

        self.prior = torch.distributions.Normal(0, 1) 

        self.pre_process = Dequantization(bits=bits)

        self.chessboard_list1 = torch.nn.ModuleList([ChessboardCoupleBlock(c, h, w, i % 2, features, hidden_blocks) for i in range(4)])
        
        h = h // 2
        w = w // 2
        c = c * 2

        self.channel_list1 = torch.nn.ModuleList([ChannelCoupleBlock(c, i % 2, features, hidden_blocks) for i in range(3)])
        self.chessboard_list2 = torch.nn.ModuleList([ChessboardCoupleBlock(c, h, w, i % 2, features, hidden_blocks) for i in range(3)])

        h = h // 2
        w = w // 2
        c = c * 2

        self.channel_list2 = torch.nn.ModuleList([ChannelCoupleBlock(c, i % 2, features, hidden_blocks) for i in range(3)])
        self.chessboard_list3 = torch.nn.ModuleList([ChessboardCoupleBlock(c, h, w, i % 2, features, hidden_blocks) for i in range(3)])

        self.out_channels = c
        self.out_h = h
        self.out_w = w

    def forward(self, x, reverse=False, average=True):
        if reverse:
            z = x
            z_chunks = []
            for i in range(2):
                chunk, z = torch.chunk(z, 2, dim=1)
                z_chunks.append(chunk)
            x = z.view(z.shape[0], self.out_channels, self.out_h, self.out_w)

            for couple in reversed(self.chessboard_list3):
                x = couple.backward(x)

            for couple in reversed(self.channel_list2):
                x = couple.backward(x)      

            x = torch.cat([x, z_chunks[-1].view(*x.shape)], dim=1)
            x = unsqueeze(x)

            for couple in reversed(self.chessboard_list2):
                x = couple.backward(x)

            for couple in reversed(self.channel_list1):
                x = couple.backward(x)

            x = torch.cat([x, z_chunks[-2].view(*x.shape)], dim=1)
            x = unsqueeze(x)

            for couple in reversed(self.chessboard_list1):
                x = couple.backward(x)  

            return self.pre_process.backward(x)
        else:
            z, LL = self.pre_process(x)

            for couple in self.chessboard_list1:
                z, ll = couple(z)
                LL = LL + ll

            z = squeeze(z)
            z, final_z = torch.chunk(z, 2, dim=1)
            final_z = final_z.view(final_z.shape[0], -1)

            for couple in self.channel_list1:
                z, ll = couple(z)
                LL = LL + ll

            for couple in self.chessboard_list2:
                z, ll = couple(z)
                LL = LL + ll

            z = squeeze(z)
            z, _z = torch.chunk(z, 2, dim=1)
            _z = _z.view(_z.shape[0], -1)
            final_z = torch.cat([final_z, _z], dim=1)        

            for couple in self.channel_list2:
                z, ll = couple(z)
                LL = LL + ll

            for couple in self.chessboard_list3:
                z, ll = couple(z)
                LL = LL + ll

            z = z.view(z.shape[0], -1)
            final_z = torch.cat([final_z, z], dim=1)
            LL = LL + torch.sum(self.prior.log_prob(final_z), dim=1)

            NLL = - LL + self.bits * self.z_size * np.log(2)

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