import numpy as np
import torch, torchvision, math
from torch.functional import F
from torch.nn import Parameter, init
import matplotlib.pyplot as plt
import pickle, torch, math, random, os
import PIL.Image as Image

class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Unflatten(torch.nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.c = c
        self.h = h
        self.w = w

    def forward(self, x):
        return x.view(-1, self.c, self.h, self.w)

class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, activation=F.relu):
        super(MLP, self).__init__()
        self.activation = activation

        self.start = torch.nn.Linear(in_features, hidden_features)

        self.middle = torch.nn.ModuleList([torch.nn.Linear(hidden_features, hidden_features) for _ in range(hidden_layers - 1)])

        self.end = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x):
        out = self.activation(self.start(x))

        for layer in self.middle:
            out = self.activation(layer(out))

        return self.end(out)

class ResBlock(torch.nn.Module):
    def __init__(self, features):
        super(ResBlock, self).__init__()
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(features, features, 1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(features, features, 3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(features, features, 1, stride=1, padding=0)
        )

    def forward(self, x):
        return x + self.shortcut(x)

class ResNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, features=256, number_blocks=8, zero_init=False):
        super(ResNet, self).__init__()
        self.in_conv = torch.nn.Conv2d(in_channels, features, 3, stride=1, padding=1)
        self.res_blocks = torch.nn.ModuleList([ResBlock(features) for _ in range(number_blocks)])
        self.out_conv = torch.nn.Conv2d(features, out_channels, 3, stride=1, padding=1)

        if zero_init:
            with torch.no_grad():
                self.out_conv.weight.fill_(0)
                self.out_conv.bias.fill_(0)

    def forward(self, x):
        h = self.in_conv(x)
        for block in self.res_blocks:
            h = block(h)
        return self.out_conv(F.relu(h, inplace=True))

class MaskLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskLinear, self).__init__(in_features, out_features)
        self.register_buffer('mask', torch.from_numpy(mask).float().t())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

class MaskConv(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.register_buffer('mask', torch.from_numpy(mask).float())

    def forward(self, input):
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)