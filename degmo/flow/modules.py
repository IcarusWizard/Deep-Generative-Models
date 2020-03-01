import numpy as np
import torch, torchvision, math
from torch.functional import F
from torch.nn import Parameter, init
import matplotlib.pyplot as plt
import pickle, torch, math, random, os
import PIL.Image as Image

from .utils import build_chessboard_mask, build_channel_mask, LOG2PI, unsqueeze, squeeze
from ..modules import Flatten, Unflatten, MLP, ResBlock, ResNet

# ------------------------- NICE ------------------------- #
class Reshape(torch.nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.flatten = Flatten()
        self.unflatten = Unflatten(c, h, w)

    def forward(self, x):
        return self.flatten(x)

    def backward(self, z):
        return self.unflatten(z)

class NICEAdditiveCoupling(torch.nn.Module):
    def __init__(self, input_dim, features, hidden_layers):
        super().__init__()
        self.coupling = MLP(input_dim // 2, input_dim // 2, features, hidden_layers)
        
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        m = self.coupling(x1)
        return torch.cat([x2 + m, x1], dim=1), 0
    
    def backward(self, z):
        z1, z2 = torch.chunk(z, 2, dim=1)
        m = self.coupling(z2)
        return torch.cat([z2, z1 - m], dim=1)

class Rescale(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.logs = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        return x * torch.exp(self.logs).view(1, -1), torch.sum(self.logs)

    def backward(self, z):
        return z * torch.exp(- self.logs).view(1, -1)

class CoupleSigmoid(torch.nn.Module):
    def __init__(self):
        super(CoupleSigmoid, self).__init__()

    def forward(self, x):
        z = torch.sigmoid(x)

        return z, torch.sum(torch.log(z + 1e-8) + torch.log(1 - z + 1e-8), dim=1)

    def backward(self, z):

        return torch.log(z) - torch.log(1 - z)

# ------------------------- RealNVP ------------------------- #
class Dequantization(torch.nn.Module):
    """
        Modified from https://github.com/fmu2/realNVP/blob/master/data_utils.py
    """
    def __init__(self, constraint=0.9, bits=8):
        super(Dequantization, self).__init__()
        self.constraint = constraint
        self.scale = 2 ** bits

    def forward(self, x):
        B, C, H, W = x.shape
        
        # dequantization
        noise = torch.rand(*x.shape, device=x.device)
        x = (x * (self.scale - 1) + noise) / self.scale
        
        # restrict data
        x *= 2.
        x -= 1.
        x *= self.constraint
        x += 1.
        x /= 2.

        # logit data
        logit_x = torch.log(x) - torch.log(1. - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(self.constraint) - np.log(1. - self.constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale) + np.log((self.scale - 1) / self.scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))

    def backward(self, z):
        x = 1. / (torch.exp(-z) + 1.) # reverse logit
        x *= 2. 
        x -= 1. 
        x /= self.constraint
        x += 1.
        x /= 2.
        return x

class AffineCoupling1D(torch.nn.Module):
    def __init__(self, input_dim, features, hidden_layers, zero_init=True):
        super().__init__()
        self.coupling = MLP(input_dim // 2, input_dim, features, hidden_layers)

        # Initialize this coupling as an identical mapping
        if zero_init:
            with torch.no_grad():
                state_dict = self.coupling.state_dict()
                state_dict['end.weight'].fill_(0)
                state_dict['end.bias'].fill_(0)
        
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        logs, t = torch.chunk(self.coupling(x1), 2, dim=1)
        logs = torch.tanh(logs)
        return torch.cat([x2 * torch.exp(logs) + t, x1], dim=1), torch.sum(logs, dim=1)
    
    def backward(self, z):
        z1, z2 = torch.chunk(z, 2, dim=1)
        logs, t = torch.chunk(self.coupling(z2), 2, dim=1)
        logs = torch.tanh(logs)
        return torch.cat([z2, (z1 - t) * torch.exp(-logs)], dim=1)

# class RealNVP_BatchNorm2D(torch.nn.Module):
#     def __init__(self, features, pho=0.9, eps=0.5):
#         super().__init__()
#         self.features = features
#         self.pho = pho
#         self.eps = eps

#         self.register_buffer('running_mean', torch.zeros(features))
#         self.register_buffer('running_var', torch.zeros(features))
#         self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
#     def forward(self, x):
#         if self.training:
#             self.num_batches_tracked += 1
#             used_mean, used_var = torch.var_mean(x, dim=(0, 2, 3))
#             cur_mean, cur_var = used_mean, used_var

#             new_mean = self.pho + self.running_mean + (1 - self.pho) * used_mean
#             new_var = self.pho * self.running_var + (1 - self.pho) * used_var

#             with torch.no_grad():
#                 self.running_mean.set_(new_mean)
#                 self.running_var.set_(new_var)

#             out_mean = new_mean / (1 - self.pho ** self.num_batches_tracked)
#             out_var = new_var / (1 - self.pho ** self.num_batches_tracked)

#         else:
#             used_mean, used_var = self.mean, self.var
#             cur_mean, cur_var = used_mean, used_var

#         return (x - out_mean.view(1, -1, 1, 1)) / torch.sqrt((out_var + self.eps)).view(1, -1, 1, 1), used_mean, out_var

class RealNVPResBlock(torch.nn.Module):
    def __init__(self, features):
        super(RealNVPResBlock, self).__init__()
        self.shortcut = torch.nn.Sequential(
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(inplace=True),
            torch.nn.utils.weight_norm(torch.nn.Conv2d(features, features, 3, stride=1, padding=1)),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(inplace=True),
            torch.nn.utils.weight_norm(torch.nn.Conv2d(features, features, 3, stride=1, padding=1)),
        )

    def forward(self, x):
        return x + self.shortcut(x)

class RealNVPResNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, features=32, number_blocks=4):
        super(RealNVPResNet, self).__init__()
        self.in_conv = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),            
            torch.nn.Conv2d(in_channels, features, 3, stride=1, padding=1),
        )

        self.res_blocks = torch.nn.ModuleList([RealNVPResBlock(features) for _ in range(number_blocks)])
        
        self.out_conv = torch.nn.Sequential(
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(inplace=True),            
            torch.nn.Conv2d(features, out_channels, 1, stride=1, padding=0),
        )

    def forward(self, x):
        h = self.in_conv(x)
        for block in self.res_blocks:
            h = block(h)
        return self.out_conv(h)

class CoupleLayers1D(torch.nn.Module):
    def __init__(self, features, hidden_features, hidden_layers, mask_type):
        super(CoupleLayers1D, self).__init__()

        mask = np.ones((1, features))
        if mask_type == 0:
            mask[0, features // 2 :] = 0
        else:
            mask[0, : features // 2] = 0
        
        mask = torch.from_numpy(mask).float()
        self.register_buffer('mask', mask)

        self.scale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        self.shift = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        self.out_bn = torch.nn.BatchNorm1d(features, affine=False)

        self.nn = MLP(features, features * 2, hidden_features, hidden_layers)

    def forward(self, x):
        logs, t = torch.chunk(self.nn(x * self.mask), 2, dim=1)
        logs = torch.tanh(logs) * self.scale + self.shift
        z = x * self.mask + (1 - self.mask) * (x * torch.exp(logs) + t)

        if self.training:
            var = torch.var(z, dim=(0), keepdim=True)
        else:
            var = self.out_bn.running_var.view(1, -1)

        z = self.out_bn(z) * (1 - self.mask) + z * self.mask

        logdet = torch.sum((logs - 0.5 * torch.log(var + self.out_bn.eps)) * (1 - self.mask), dim=1)

        return z, logdet

    def backward(self, z):
        mean, var, eps = self.out_bn.running_mean, self.out_bn.running_var, self.out_bn.eps
        z = z * torch.sqrt(var + eps).view(1, -1) + mean.view(1, -1)

        logs, t = torch.chunk(self.nn(z * self.mask), 2, dim=1)
        logs = torch.tanh(logs) * self.scale + self.shift
        x = z * self.mask + (1 - self.mask) * (z - t) * torch.exp(- logs)

        return x

class CoupleLayer2D(torch.nn.Module):
    def __init__(self, in_channels, features=32, number_blocks=4):
        super(CoupleLayer2D, self).__init__()

        self.res = RealNVPResNet(in_channels, 2 * in_channels, features, number_blocks)
        self.scale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        self.shift = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        self.out_bn = torch.nn.BatchNorm2d(in_channels, affine=False)
        self.build_mask()
        
    def forward(self, x):
        logs, t = torch.chunk(self.res(x * self.mask), 2, dim=1)
        logs = torch.tanh(logs) * self.scale + self.shift

        logs = logs * (1 - self.mask)
        t = t * (1 - self.mask)

        z = x * torch.exp(logs) + t

        if self.training:
            var = torch.var(z, dim=(0, 2, 3)).view(1, -1, 1, 1)
        else:
            var = self.out_bn.running_var.view(1, -1, 1, 1)

        z = self.out_bn(z) * (1 - self.mask) + z * self.mask

        logdet = torch.sum(logs - 0.5 * torch.log(var + self.out_bn.eps) * (1 - self.mask), dim=(1, 2, 3))

        return z, logdet

    def backward(self, z):
        mean, var, eps = self.out_bn.running_mean, self.out_bn.running_var, self.out_bn.eps
        z = z * torch.exp(0.5 * torch.log(var.view(1, -1, 1, 1) + eps) * (1. - self.mask)) + mean.view(1, -1, 1, 1) * (1. - self.mask)

        logs, t = torch.chunk(self.res(z * self.mask), 2, dim=1)
        logs = torch.tanh(logs) * self.scale + self.shift
        x = z * self.mask + (1 - self.mask) * (z - t) * torch.exp(- logs)

        return x

    def build_mask(self):
        """
            This function need to set self.mask to the correct couple mask
        """
        raise NotImplementedError

class ChessboardCoupleLayer(CoupleLayer2D):
    def __init__(self, in_channels, h, w, inverse=False, features=32, number_blocks=4):
        self.inverse = inverse
        self.h = h
        self.w = w
        super(ChessboardCoupleLayer, self).__init__(in_channels, features, number_blocks)

    def build_mask(self):
        mask = build_chessboard_mask(self.h, self.w)
        if self.inverse:
            mask = 1 - mask
        mask = torch.from_numpy(mask).float().view(1, 1, self.h, self.w)
        self.register_buffer('mask', mask)

class ChannelCoupleLayer(CoupleLayer2D):
    def __init__(self, in_channels, inverse=False, features=32, number_blocks=4):
        self.inverse = inverse
        self.in_channels = in_channels
        super(ChannelCoupleLayer, self).__init__(in_channels, features, number_blocks)

    def build_mask(self):
        mask = build_channel_mask(self.in_channels)
        if self.inverse:
            mask = 1 - mask
        mask = torch.from_numpy(mask).float().view(1, self.in_channels, 1, 1)
        self.register_buffer('mask', mask)

class MultiLayerBlock(torch.nn.Module):
    def __init__(self, c, h, w, features=32, number_blocks=4):
        super().__init__()

        self.chessboard_list = torch.nn.ModuleList([
            ChessboardCoupleLayer(c, h, w, i % 2, features, number_blocks) for i in range(3)
        ])
        
        h = h // 2
        w = w // 2
        c = c * 2

        self.channel_list = torch.nn.ModuleList([
            ChannelCoupleLayer(c, i % 2, features, number_blocks) for i in range(3)
        ])

    def forward(self, x):
        z = x
        logdet = 0

        for coupling in self.chessboard_list:
            z, _logdet = coupling(z)
            logdet = logdet + _logdet

        z = squeeze(z)
        z, final_z = torch.chunk(z, 2, dim=1)

        for coupling in self.channel_list:
            z, _logdet = coupling(z)
            logdet = logdet + _logdet

        return z, final_z, logdet

    def backward(self, z, final_z):
        x = z

        for coupling in reversed(self.channel_list):
            x = coupling.backward(x)

        x = torch.cat([x, final_z.view(*x.shape)], dim=1)
        x = unsqueeze(x)

        for coupling in reversed(self.chessboard_list):
            x = coupling.backward(x)

        return x        

# ------------------------- Glow ------------------------- #
class ActNorm1D(torch.nn.Module):
    def __init__(self, features):
        super(ActNorm1D, self).__init__()

        self.weight = torch.nn.Parameter(torch.zeros(features))

        self.bias = torch.nn.Parameter(torch.zeros(features))

        self.register_buffer('is_init', torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        if not self.is_init:
            mean = torch.mean(x, dim=0).detach()
            var = torch.sqrt(torch.var(x, dim=0)).detach()

            with torch.no_grad():
                self.weight.copy_(- torch.log(var))
                self.bias.copy_(- mean / var)

            self.is_init = 1 - self.is_init

        scale = torch.exp(self.weight)

        return x * scale.view(1, -1) + self.bias.view(1, -1), torch.sum(self.weight)

    def backward(self, z):
        scale = torch.exp(self.weight)

        return (z - self.bias.view(1, -1)) * scale.view(1, -1).reciprocal()

class ActNorm2D(torch.nn.Module):
    def __init__(self, features):
        super(ActNorm2D, self).__init__()
        self.features = features

        self.weight = torch.nn.Parameter(torch.zeros(features))

        self.bias = torch.nn.Parameter(torch.zeros(features))

        self.register_buffer('is_init', torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        if not self.is_init:
            mean = torch.mean(x.permute(1, 0, 2, 3).contiguous().view(self.features, -1), dim=1).detach()
            var = torch.sqrt(torch.var(x.permute(1, 0, 2, 3).contiguous().view(self.features, -1), dim=1)).detach()

            with torch.no_grad():
                self.weight.copy_(- torch.log(var))
                self.bias.copy_(- mean / var)

            self.is_init = 1 - self.is_init

        scale = torch.exp(self.weight)
        hw = np.prod(x.shape[2:])

        return x * scale.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1), hw * torch.sum(self.weight)

    def backward(self, z):
        scale = torch.exp(self.weight)

        return (z - self.bias.view(1, -1, 1, 1)) * scale.view(1, -1, 1, 1).reciprocal()

class SimpleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, features=512, zero_init=True):
        super(SimpleConv, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, features, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(features, features, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(features, out_channels, 3, stride=1, padding=1)

        if zero_init:
            with torch.no_grad():
                self.conv3.weight.fill_(0)
                self.conv3.bias.fill_(0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return self.conv3(x)

class Invertible1x1Conv(torch.nn.Module):
    def __init__(self, channels, use_lu=True):
        super(Invertible1x1Conv, self).__init__()
        self.channels = channels
        self.use_lu = use_lu

        if self.use_lu:
            from scipy import linalg
            np_w = linalg.qr(np.random.randn(channels, channels))[0].astype('float32')

            np_p, np_l, np_u = linalg.lu(np_w)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)
            mask = np.tril(np.ones((channels, channels)), -1) # mask for left triangle

            self.register_buffer('p', torch.from_numpy(np_p).float())
            self.register_buffer('sign_s', torch.from_numpy(np_sign_s).float())
            self.register_buffer('mask', torch.from_numpy(mask).float())
            self.l = torch.nn.Parameter(torch.from_numpy(np_l).float())
            self.logs = torch.nn.Parameter(torch.from_numpy(np_log_s).float())
            self.u = torch.nn.Parameter(torch.from_numpy(np_u).float())
        else:
            w_init = np.linalg.qr(np.random.randn(channels, channels))[0].astype('float32')

            self.w = torch.nn.Parameter(torch.from_numpy(w_init).float())

    def forward(self, x):
        if self.use_lu:
            eye = torch.eye(self.channels, dtype=x.dtype, device=x.device)
            l = self.l * self.mask + eye
            u = self.u * self.mask.T + torch.diag(self.sign_s * torch.exp(self.logs))
            w = self.p @ l @ u

            logdet = torch.sum(self.logs) * np.prod(x.shape[2:])
        else:
            w = self.w
            logdet = torch.log(torch.abs(torch.det(self.w)) + 1e-12)

        z = F.conv2d(x, w.view(self.channels, self.channels, 1, 1))

        return z, logdet

    def backward(self, z):
        if self.use_lu:
            eye = torch.eye(self.channels, dtype=z.dtype, device=z.device)
            l = self.l * self.mask + eye
            u = self.u * self.mask.T + torch.diag(self.sign_s * torch.exp(self.logs))
            w = self.p @ l @ u

            inv_l = torch.inverse(l)
            inv_u = torch.inverse(u)
            inv_p = torch.inverse(self.p)

            inv_w = inv_u @ inv_l @ inv_p
        else:
            inv_w = torch.inverse(self.w)

        x = F.conv2d(z, inv_w.view(self.channels, self.channels, 1, 1))

        return x

class GlowAffineLayer2D(torch.nn.Module):
    def __init__(self, in_channels, features=512):
        super(GlowAffineLayer2D, self).__init__()
        self.in_channels = in_channels
        self.res = SimpleConv(in_channels // 2, in_channels, features, zero_init=True)
        
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        logs, t = torch.chunk(self.res(x2), 2, dim=1) # NOTE: 
        # NOTE: refer to https://github.com/openai/glow/blob/eaff2177693a5d84a1cf8ae19e8e0441715b82f8/model.py#L395
        # Without constrain, logs > 88.8 can cause torch.exp(logs) to inf. Original implementation here use 
        # scale = sigmoid(logs + 2), here we use scale = exp(tanh(logs)) to consist with other models
        logs = torch.tanh(logs)

        z1 = x1 * torch.exp(logs) + t
        z = torch.cat([z1, x2], dim=1)

        return z, torch.sum(logs, dim=(1, 2, 3))

    def backward(self, z):
        z1, z2 = torch.chunk(z, 2, dim=1)
        logs, t = torch.chunk(self.res(z2), 2, dim=1)
        logs = torch.tanh(logs)
        
        x1 = (z1 - t) * torch.exp(- logs)
        x = torch.cat([x1, z2], dim=1)

        return x

class GlowAdditiveLayer2D(torch.nn.Module):
    def __init__(self, in_channels, features=512):
        super(GlowAdditiveLayer2D, self).__init__()
        self.in_channels = in_channels
        self.res = SimpleConv(in_channels // 2, in_channels // 2, features, zero_init=True)
        
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        t = self.res(x2)

        z1 = x1 + t
        z = torch.cat([z1, x2], dim=1)

        return z, 0

    def backward(self, z):
        z1, z2 = torch.chunk(z, 2, dim=1)
        t = self.res(z2)
        
        x1 = z1 - t
        x = torch.cat([x1, z2], dim=1)

        return x

class GlowAffineStep(torch.nn.Module):
    def __init__(self, channels, features=512, use_lu=True):
        super(GlowAffineStep, self).__init__()

        self.norm = ActNorm2D(channels)
        self.permute = Invertible1x1Conv(channels, use_lu)
        self.couple = GlowAffineLayer2D(channels, features)

    def forward(self, x):
        z, logdet = self.norm(x)

        # NOTE: The permute step will shuffle the data in channels, so although we has normalized
        # the data with ActNorm, the permute operation will keep mixing the data from different
        # channels and the logdet will not stay at zero until the whole data is normalized.
        z, _logdet = self.permute(z)
        logdet = logdet + _logdet

        z, _logdet = self.couple(z)
        logdet = logdet + _logdet

        return z, logdet

    def backward(self, z):
        x = self.couple.backward(z)
        x = self.permute.backward(x)
        x = self.norm.backward(x)

        return x

class GlowAdditiveStep(torch.nn.Module):
    def __init__(self, channels, features=512, use_lu=True):
        super(GlowAdditiveStep, self).__init__()

        self.norm = ActNorm2D(channels)
        self.permute = Invertible1x1Conv(channels, use_lu)
        self.couple = GlowAdditiveLayer2D(channels, features)

    def forward(self, x):
        z, logdet = self.norm(x)

        # NOTE: The permute step will shuffle the data in channels, so although we has normalized
        # the data with ActNorm, the permute operation will keep mixing the data from different
        # channels and the logdet will not stay at zero until the whole data is normalized.
        z, _logdet = self.permute(z)
        logdet = logdet + _logdet

        z, _logdet = self.couple(z)
        logdet = logdet + _logdet

        return z, logdet

    def backward(self, z):
        x = self.couple.backward(z)
        x = self.permute.backward(x)
        x = self.norm.backward(x)

        return x