import torch
from torch.functional import F
import numpy as np

from .modules import ConvGenerator, ConvDiscriminator, MLPGenerator, MLPDiscriminator
from .utils import weights_init

class WGAN(torch.nn.Module):
    def __init__(self, c, h, w, mode, latent_dim,
        discriminator_features, discriminator_hidden_layers,
        generator_features, generator_hidden_layers,
        use_norm_discriminator, use_norm_generator):
        super().__init__()

        self.input_size = (c, h, w)
        self.latent_dim = latent_dim

        if mode == 'mlp':
            self.generator = MLPGenerator(c, h, w, latent_dim, generator_features, generator_hidden_layers)

            self.discriminator = MLPDiscriminator(c, h, w, discriminator_features, discriminator_hidden_layers)
            
        elif mode == 'conv':
            assert h == w, "only support square images"

            downsampling = int(np.ceil(np.log2(h))) - 2

            self.criterion = torch.nn.BCEWithLogitsLoss()

            self.generator = ConvGenerator(c, h, w, latent_dim, generator_features, use_norm_generator)

            self.discriminator = ConvDiscriminator(c, h, w, discriminator_features, use_norm_discriminator)

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
        return torch.mean(self.discriminate(real)) - torch.mean(self.discriminate(fake))

    def get_generator_loss(self, fake):
        return torch.mean(self.discriminate(fake))