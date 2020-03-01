import torch
from torch.functional import F
import numpy as np

from degmo.flow.distribution import FlowDistribution1D
from ..modules import MLP, Flatten, Unflatten
from .utils import get_kl, LOG2PI

class FVAE(torch.nn.Module):
    def __init__(self, c=3, h=32, w=32, latent_dim=2, features=128, hidden_layers=5, 
                 flow_hidden_layers=3, flow_features=64, flow_num_transformation=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = c * h * w

        self.encoder = torch.nn.Sequential(
            Flatten(),
            MLP(self.input_dim, 2 * latent_dim, features, hidden_layers),
        )

        output_dim = self.input_dim * 2 
        output_c = 2 * c
        
        self.decoder = torch.nn.Sequential(
            MLP(latent_dim, output_dim, features, hidden_layers),
            Unflatten(output_c, h, w),
        )
        
        self.prior = FlowDistribution1D(latent_dim, flow_num_transformation, flow_features, flow_hidden_layers)

        self.init_prior = torch.distributions.Normal(0, 1)

    def forward(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)

        # reparameterize trick
        epsilon = torch.randn_like(logs)
        z = mu + epsilon * torch.exp(logs)

        # Use Mento Carlo Estimation to compute kl divergence
        # kl = log q_{\phi}(z|x) - log p_{\theta}(z)
        post_prob = torch.sum(- epsilon ** 2 / 2 - LOG2PI - logs, dim=1) 
        prior_prob = self.prior.log_prob(z)
        init_prior_prob = torch.sum(self.init_prior.log_prob(z), dim=1)

        kl = post_prob - prior_prob

        _x = self.decoder(z)

        # output is a gauss with diagonal variance
        _mu, _logs = torch.chunk(_x, 2, dim=1)
        _logs = torch.tanh(_logs)
        reconstruction_loss = (x - _mu) ** 2 / 2 * torch.exp(-2 * _logs) + LOG2PI + _logs


        kl = torch.mean(kl)
        reconstruction_loss = torch.mean(torch.sum(reconstruction_loss, dim=(1, 2, 3)))
        extra_info = torch.mean(prior_prob - init_prior_prob) # D_KL(p_{\theta} || p_{\theta_{init}})

        return kl + reconstruction_loss, {
            "KL divergence" : kl,
            "Reconstruction loss" : reconstruction_loss,
            "Extra information" : extra_info,
        }
    
    def encode(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)
        return mu

    def decode(self, z, deterministic=True):
        _x = self.decoder(z)

        _mu, _logs = torch.chunk(_x, 2, dim=1)
        _logs = torch.tanh(_logs)
        x = _mu
        if not deterministic:
            x = x + torch.exp(_logs) * torch.randn_like(x)

        return x

    def sample(self, number=1000, deterministic=True):

        z = self.prior.sample(num=number)

        return self.decode(z, deterministic=deterministic)