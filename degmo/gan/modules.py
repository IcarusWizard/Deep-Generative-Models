import numpy as np
import torch, torchvision, math
from torch.functional import F
from torch.nn import Parameter, init
from torch import nn
import matplotlib.pyplot as plt
import pickle, torch, math, random, os
import PIL.Image as Image
from functools import partial

from ..modules import Flatten, Unflatten, MLP

class MaxOut(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear = torch.nn.Linear(in_channel, 2 * out_channel)

    def forward(self, x):
        out = self.linear(x)
        return torch.max(*torch.chunk(x, 2, dim=1))

class MLPGenerator(torch.nn.Module):
    def __init__(self, c, h, w, latent_dim, features, layers):
        super().__init__()

        self.input_size = c * h * w
        self.latent_dim = latent_dim

        self.generator = torch.nn.Sequential(
            MLP(latent_dim, self.input_size, features, layers),
            Unflatten([c, h, w]),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.generator(x)

class MLPDiscriminator(torch.nn.Module):
    def __init__(self, c, h, w, features, layers):
        super().__init__()

        self.input_size = c * h * w

        self.discriminator = torch.nn.Sequential(
            Flatten(),
            MLP(self.input_size, 1, features, layers, partial(torch.nn.LeakyReLU, negative_slope=0.01)),
        )

    def forward(self, x):
        return self.discriminator(x)   

class ConvGenerator(torch.nn.Module):
    def __init__(self, c, h, w, latent_dim, features, use_norm=True):
        super().__init__()

        downsampling = int(np.ceil(np.log2(h))) - 2

        generator = []
        out_features = features

        generator.append(torch.nn.Linear(latent_dim, out_features * 4 * 4, bias=False))
        generator.append(Unflatten([out_features, 4, 4]))
        
        for i in range(downsampling - 1):
            if use_norm:
                generator.append(torch.nn.BatchNorm2d(out_features))
            generator.append(torch.nn.ReLU(True))
            if i == 0 and h == 28: # mnist
                generator.append(torch.nn.ConvTranspose2d(out_features, out_features // 2, 4, 
                    stride=2, padding=2, output_padding=1, bias=False))
            else:
                generator.append(torch.nn.ConvTranspose2d(out_features, out_features // 2, 4, stride=2, padding=1, bias=False))

            out_features = out_features // 2

        if use_norm:
            generator.append(torch.nn.BatchNorm2d(out_features))
        generator.append(torch.nn.ReLU(True))
        generator.append(torch.nn.ConvTranspose2d(out_features, c, 4, stride=2, padding=1, bias=False))
        generator.append(torch.nn.Tanh())

        self.generator = torch.nn.Sequential(*generator)

    def forward(self, x):
        return self.generator(x)

class ConvDiscriminator(torch.nn.Module):
    def __init__(self, c, h, w, features, use_norm=True):
        super().__init__()

        downsampling = int(np.ceil(np.log2(h))) - 2    

        discriminator = []
        in_features = c
        out_features = features // (2 ** (downsampling - 1))
        
        for i in range(downsampling):
            if i == downsampling - 1 and h == 28: # mnist
                discriminator.append(torch.nn.Conv2d(in_features, out_features, 4, stride=2, padding=2, bias=False))
            else:
                discriminator.append(torch.nn.Conv2d(in_features, out_features, 4, stride=2, padding=1, bias=False))
            if use_norm:
                discriminator.append(torch.nn.BatchNorm2d(out_features))
            discriminator.append(torch.nn.LeakyReLU(0.2, True))

            in_features = out_features
            out_features *= 2

        discriminator.append(Flatten())
        discriminator.append(torch.nn.Linear(in_features * 4 * 4, 1, bias=False))

        self.discriminator = torch.nn.Sequential(*discriminator)

    def forward(self, x):
        return self.discriminator(x)

class UpsampleConv2d(torch.nn.Module):
    def __init__(self, input_dim, output_dim=256, filter_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_dim, output_dim, filter_size, padding=(filter_size // 2))

    def forward(self, x):
        _x = torch.cat([x, x, x, x], dim=1)
        _x = F.pixel_shuffle(_x, 2)
        return self.conv(_x)

class DownsampleConv2d(torch.nn.Module):
    def __init__(self, input_dim, output_dim=256, filter_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_dim, output_dim, filter_size, padding=(filter_size // 2))

    def forward(self, x):
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C*4, H//2, W//2)
        x = torch.mean(x.view(B, 4, C, H//2, W//2), dim=1)
        return self.conv(x)

class ResBlockUp(torch.nn.Module):
    def __init__(self, input_dim, output_dim, filter_size=3):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(input_dim)
        self.conv = torch.nn.Conv2d(input_dim, output_dim, filter_size, padding=(filter_size // 2))
        self.bn2 = torch.nn.BatchNorm2d(output_dim)

        self.residual = UpsampleConv2d(output_dim, output_dim, filter_size)
        self.shortcut = UpsampleConv2d(input_dim, output_dim, 1)

    def forward(self, x):
        _x = F.relu(self.bn1(x))
        _x = F.relu(self.bn2(self.conv(_x)))

        return self.residual(_x) + self.shortcut(x)

class ResBlockDown(torch.nn.Module):
    def __init__(self, input_dim, output_dim, filter_size):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_dim, output_dim, filter_size, padding=(filter_size // 2))

        self.residual = DownsampleConv2d(output_dim, output_dim, filter_size)
        self.shortcut = DownsampleConv2d(input_dim, output_dim, 1)

    def forward(self, x):
        _x = F.relu(x)
        _x = F.relu(self.conv(_x))

        return self.residual(_x) + self.shortcut(x)

class ResBlockGenerator(torch.nn.Module):
    def __init__(self, filter_num=128, filter_size=3):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(filter_num)
        self.conv = torch.nn.Conv2d(filter_num, filter_num, filter_size, padding=(filter_size // 2))
        self.bn2 = torch.nn.BatchNorm2d(filter_num)

    def forward(self, x):
        _x = F.relu(self.bn1(x))
        _x = F.relu(self.bn2(self.conv(_x)))

        return x + _x

class ResBlockDiscriminator(torch.nn.Module):
    def __init__(self, filter_num=128, filter_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(filter_num, filter_num, filter_size, padding=(filter_size // 2))

    def forward(self, x):
        _x = F.relu(x)
        _x = F.relu(self.conv(_x))

        return x + _x

class ResGenerator(torch.nn.Module):
    def __init__(self, c, h, w, latent_dim=128, features=128):
        super().__init__()
        self.latent_dim = latent_dim

        downsampling = int(np.ceil(np.log2(h))) - 2

        self.dense = torch.nn.Linear(latent_dim, 4 * 4 * features)
        self.upblocks = torch.nn.ModuleList([ResBlockUp(features, features, 3) for i in range(downsampling)])

        self.output_bn = torch.nn.BatchNorm2d(features)
        self.output_conv = torch.nn.Conv2d(features, 3, 3, padding=1)

    def forward(self, z):
        space_z = self.dense(z).view(z.shape[0], -1, 4, 4)

        for upblock in self.upblocks:
            space_z = upblock(space_z)

        out = F.relu(self.output_bn(space_z))
        return torch.tanh(self.output_conv(out))

class ResDiscriminator(torch.nn.Module):
    def __init__(self, c, h, w, features=128, hidden_layers=2):
        super().__init__()
        downsampling = int(np.ceil(np.log2(h))) - 3
        
        downblocks = [ResBlockDown(3, features, 3)]
        for i in range(downsampling - 1):
            downblocks.append(ResBlockDown(features, features, 3))

        self.downblocks = torch.nn.ModuleList(downblocks)

        self.resblocks = torch.nn.ModuleList([ResBlockDiscriminator(features, 3) for _ in range(hidden_layers)])

        self.dense = torch.nn.Linear(features, 1)

    def forward(self, x):
        _x = x

        for downblock in self.downblocks:
            _x = downblock(_x)

        for resblock in self.resblocks:
            _x = resblock(_x)

        _x = F.relu(_x)
        _x = torch.mean(_x, dim=(2, 3))

        return self.dense(_x)

class SelfAttention(nn.Module):
    def __init__(self, features):
        super().__init__()
        
        self.query_conv = torch.nn.Conv2d(features , features // 8, 1)
        self.key_conv = torch.nn.Conv2d(features , features // 8, 1)
        self.value_conv = torch.nn.Conv2d(features , features, 1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        proj_query  = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key =  self.key_conv(x).view(b, -1, h * w)
        energy =  torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=2) 
        proj_value = self.value_conv(x).view(b, -1, h * w)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        
        out = self.gamma*out + x
        return out

class SelfAttentionGenerator(torch.nn.Module):
    def __init__(self, c, h, w, latent_dim=128, features=128):
        super().__init__()
        self.latent_dim = latent_dim

        downsampling = int(np.ceil(np.log2(h))) - 2

        self.dense = torch.nn.Linear(latent_dim, 4 * 4 * features)

        upblocks = [ResBlockUp(features, features, 3) for i in range(downsampling)]
        upblocks.insert(-1, SelfAttention(features))
        self.upblocks = torch.nn.ModuleList(upblocks)

        self.output_bn = torch.nn.BatchNorm2d(features)
        self.output_conv = torch.nn.Conv2d(features, 3, 3, padding=1)

    def forward(self, z):
        space_z = self.dense(z).view(z.shape[0], -1, 4, 4)

        for upblock in self.upblocks:
            space_z = upblock(space_z)

        out = F.relu(self.output_bn(space_z))
        return torch.tanh(self.output_conv(out))

class SelfAttentionDiscriminator(torch.nn.Module):
    def __init__(self, c, h, w, features=128, hidden_layers=2):
        super().__init__()
        downsampling = int(np.ceil(np.log2(h))) - 3
        
        downblocks = [ResBlockDown(3, features, 3), SelfAttention(features)]
        for i in range(downsampling - 1):
            downblocks.append(ResBlockDown(features, features, 3))

        self.downblocks = torch.nn.ModuleList(downblocks)

        self.resblocks = torch.nn.ModuleList([ResBlockDiscriminator(features, 3) for _ in range(hidden_layers)])

        self.dense = torch.nn.Linear(features, 1)

    def forward(self, x):
        _x = x

        for downblock in self.downblocks:
            _x = downblock(_x)

        for resblock in self.resblocks:
            _x = resblock(_x)

        _x = F.relu(_x)
        _x = torch.sum(_x, dim=(2, 3))

        return self.dense(_x)