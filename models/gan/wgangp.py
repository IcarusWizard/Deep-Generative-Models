import torch
from torch.functional import F
import numpy as np

from .modules import ConvGenerator, ConvDiscriminator, ResGenerator, ResDiscriminator
from .utils import weights_init

class WGANGP(torch.nn.Module):
    def __init__(self, c, h, w, mode, latent_dim, _lambda,
        discriminator_features, discriminator_hidden_layers, generator_features):
        super().__init__()
        assert h == w, "only support square images"

        self.input_size = (c, h, w)
        self.latent_dim = latent_dim
        self._lambda = _lambda

        if mode == 'res':
            self.generator = ResGenerator(c, h, w, latent_dim, generator_features)

            self.discriminator = ResDiscriminator(c, h, w, discriminator_features, discriminator_hidden_layers)
            
        elif mode == 'conv':
            self.generator = ConvGenerator(c, h, w, latent_dim, generator_features, True)

            self.discriminator = ConvDiscriminator(c, h, w, discriminator_features, False)

            self.generator.generator.apply(weights_init)
            self.discriminator.discriminator.apply(weights_init)

    def generate(self, samples):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        z = torch.randn(samples, self.latent_dim, dtype=dtype, device=device)

        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def z2x(self, z):
        return self.generator(z)

    def get_discriminator_loss(self, real, fake):
        t = torch.rand_like(real)
        x_prime = real * t + (1 - t) * fake

        score = self.discriminator(x_prime)

        grad = torch.autograd.grad(score, x_prime, torch.ones_like(score), create_graph=True, retain_graph=True)[0]
        grad = grad.view(grad.shape[0], -1)

        return torch.mean(self.discriminate(real)) - torch.mean(self.discriminate(fake)) \
            + self._lambda * torch.mean((torch.sqrt(torch.sum(grad ** 2, dim=1)) - 1) ** 2)

    def get_generator_loss(self, fake):
        return torch.mean(self.discriminate(fake))