import torch
from torch.functional import F
import numpy as np

from ..modules import MLP, Flatten, Unflatten, ResNet
from .utils import get_kl, LOG2PI

class CONV_VAE(torch.nn.Module):
    def __init__(self, c=3, h=32, w=32, latent_dim=16, down_sampling=2, features=256, res_layers=5, 
                 output_type='fix_std', use_mce=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_type = output_type
        self.use_mce = use_mce

        self.feature_shape = (features, h // (2 ** down_sampling), w // (2 ** down_sampling))

        encoder_list = []
        for i in range(down_sampling):
            encoder_list.append(torch.nn.Conv2d(c if i == 0 else features, features, 4, 2, padding=1))
            encoder_list.append(torch.nn.ReLU())
        
        encoder_list.append(ResNet(features, features, features, res_layers))
        encoder_list.append(Flatten())
        encoder_list.append(torch.nn.Linear(np.prod(self.feature_shape), 2 * latent_dim))

        self.encoder = torch.nn.Sequential(*encoder_list)

        decoder_list = []
        decoder_list.append(torch.nn.Linear(latent_dim, np.prod(self.feature_shape)))
        decoder_list.append(Unflatten(*self.feature_shape))

        for i in range(down_sampling):
            decoder_list.append(torch.nn.ConvTranspose2d(features, features, 4, 2, padding=1))
            decoder_list.append(torch.nn.ReLU())
        
        output_c = 2 * c if self.output_type == 'gauss' else c
        decoder_list.append(torch.nn.Conv2d(features, output_c, 3, stride=1, padding=1))
        
        self.decoder = torch.nn.Sequential(*decoder_list)
        
        self.prior = torch.distributions.Normal(0, 1)

    def forward(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)

        # reparameterize trick
        epsilon = torch.randn(*logs.shape, device=x.device)
        z = mu + epsilon * torch.exp(logs)

        # compute kl divergence
        if self.use_mce: # Use Mento Carlo Estimation
            # kl = log q_{\phi}(z|x) - log p_{\theta}(z)
            kl = torch.sum(- epsilon ** 2 / 2 - LOG2PI - logs - self.prior.log_prob(z), dim=1)
        else:
            kl = get_kl(mu, logs)

        _x = self.decoder(z)

        # compute reconstruction loss
        if self.output_type == 'fix_std':
            # output is a gauss with a fixed 1 variance,
            # reconstruction loss is mse plus constant
            reconstruction_loss = (x - _x) ** 2 / 2 + LOG2PI

        elif self.output_type == 'gauss':
            # output is a gauss with diagonal variance
            _mu, _logs = torch.chunk(_x, 2, dim=1)
            _logs = torch.tanh(_logs)
            reconstruction_loss = (x - _mu) ** 2 / 2 * torch.exp(-2 * _logs) + LOG2PI + _logs

        elif self.output_type == 'bernoulli':
            # output is the logit of a bernouli distribution,
            # reconstruction loss is cross-entropy
            p = torch.sigmoid(_x)
            reconstruction_loss = - x * torch.log(p + 1e-8) - (1 - x) * torch.log(1 - p + 1e-8)

        kl = torch.mean(kl)
        reconstruction_loss = torch.mean(torch.sum(reconstruction_loss, dim=(1, 2, 3)))

        return kl + reconstruction_loss, {
            "KL divergence" : kl,
            "Reconstruction loss" : reconstruction_loss,
        }
    
    def encode(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)
        return mu

    def decode(self, z, deterministic=True):
        _x = self.decoder(z)

        if self.output_type == 'fix_std':
            x = _x
            if not deterministic:
                x = x + torch.randn_like(x)

        elif self.output_type == 'gauss':
            _mu, _logs = torch.chunk(_x, 2, dim=1)
            _logs = torch.tanh(_logs)
            x = _mu
            if not deterministic:
                x = x + torch.exp(_logs) * torch.randn_like(x)

        elif self.output_type == 'bernoulli':
            p = torch.sigmoid(_x)
            if not deterministic:
                x = (torch.rand_like(p) < p).float()
            else:
                x = (p > 0.5).float()

        return x

    def sample(self, number=1000, deterministic=True):
        device = next(self.parameters()).device

        z = torch.randn(number, self.latent_dim, device=device)

        return self.decode(z, deterministic=deterministic)