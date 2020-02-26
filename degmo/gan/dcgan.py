import torch
from torch.functional import F
import numpy as np

from .modules import ConvGenerator, ConvDiscriminator
from .utils import weights_init

class DCGAN(torch.nn.Module):
    def __init__(self, c, h, w, latent_dim, discriminator_features, generator_features):
        super().__init__()

        self.input_size = (c, h, w)
        self.out_h = 4
        self.out_w = 4
        self.latent_dim = latent_dim

        assert h == w, "only support square images"

        downsampling = int(np.ceil(np.log2(h))) - 2

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.generator = ConvGenerator(c, h, w, latent_dim, generator_features)

        self.discriminator = ConvDiscriminator(c, h, w, discriminator_features)

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
        real_logit = self.discriminate(real)
        fake_logit = self.discriminate(fake)

        fake_label = torch.zeros_like(fake_logit)
        real_label = torch.ones_like(real_logit)

        return self.criterion(real_logit, real_label) + self.criterion(fake_logit, fake_label)

    def get_generator_loss(self, fake):
        fake_logit = self.discriminate(fake)

        fake_label = torch.ones_like(fake_logit)

        return self.criterion(fake_logit, fake_label)