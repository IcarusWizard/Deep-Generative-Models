import torch
from torch.functional import F
import numpy as np

from .modules import SelfAttentionGenerator, SelfAttentionDiscriminator
from .utils import add_sn

class SAGAN(torch.nn.Module):
    def __init__(self, c, h, w, latent_dim,
        discriminator_features, discriminator_hidden_layers, generator_features):
        super().__init__()
        assert h == w, "only support square images"

        self.input_size = (c, h, w)
        self.latent_dim = latent_dim

        self.generator = SelfAttentionGenerator(c, h, w, latent_dim, generator_features)

        self.discriminator = SelfAttentionDiscriminator(c, h, w, discriminator_features, discriminator_hidden_layers)

        self.discriminator.apply(add_sn)
        self.generator.apply(add_sn)

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
        real_score = self.discriminate(real)
        fake_score = self.discriminate(fake)

        return - torch.mean(torch.min(torch.zeros_like(real_score), -1 + real_score)) \
            - torch.mean(torch.min(torch.zeros_like(fake_score), -1 - fake_score))

    def get_generator_loss(self, fake):
        fake_score = self.discriminate(fake)

        return - torch.mean(fake_score)