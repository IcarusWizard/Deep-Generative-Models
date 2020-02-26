import torch
from torch.functional import F
import numpy as np

from .modules import ConvGenerator, ConvDiscriminator, ResGenerator, ResDiscriminator
from .utils import weights_init, add_sn

class SNGAN(torch.nn.Module):
    def __init__(self, c, h, w, mode, latent_dim,
        discriminator_features, discriminator_hidden_layers, generator_features):
        super().__init__()
        assert h == w, "only support square images"

        self.input_size = (c, h, w)
        self.latent_dim = latent_dim
        self.criterion = torch.nn.BCEWithLogitsLoss()

        if mode == 'res':
            self.generator = ResGenerator(c, h, w, latent_dim, generator_features)

            self.discriminator = ResDiscriminator(c, h, w, discriminator_features, discriminator_hidden_layers)
            
        elif mode == 'conv':
            self.generator = ConvGenerator(c, h, w, latent_dim, generator_features, True)

            self.discriminator = ConvDiscriminator(c, h, w, discriminator_features, False)

            self.generator.generator.apply(weights_init)
            self.discriminator.discriminator.apply(weights_init)

        self.discriminator.apply(add_sn)

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
        real_logit = self.discriminate(real)
        fake_logit = self.discriminate(fake)

        fake_label = torch.zeros_like(fake_logit)
        real_label = torch.ones_like(real_logit)

        return self.criterion(real_logit, real_label) + self.criterion(fake_logit, fake_label)

    def get_generator_loss(self, fake):
        fake_logit = self.discriminate(fake)

        fake_label = torch.ones_like(fake_logit)

        return self.criterion(fake_logit, fake_label)