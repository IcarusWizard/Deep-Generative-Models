import numpy as np
import torch, torchvision, math
from torch.functional import F
from torch.nn import Parameter, init
import matplotlib.pyplot as plt
import pickle, torch, math, random, os
import PIL.Image as Image

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
            Unflatten(c, h, w),
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
            MLP(self.input_size, 1, features, layers, lambda x: F.leaky_relu(x, 0.01)),
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
        generator.append(Unflatten(out_features, 4, 4))
        
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
