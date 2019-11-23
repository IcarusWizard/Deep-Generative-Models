import torch
from torch.functional import F

from ..modules import Flatten, Unflatten, MLP
from .modules import MaxOut

class GAN(torch.nn.Module):
    def __init__(self, c, h, w, latent_dim, 
        discriminator_features, discriminator_hidden_layers,
        generator_features, generator_hidden_layers):
        super().__init__()

        self.input_size = c * h * w
        self.latent_dim = latent_dim

        self.generator = torch.nn.Sequential(
            MLP(latent_dim, self.input_size, generator_features, generator_hidden_layers),
            Unflatten(c, h, w),
            torch.nn.Sigmoid(),
        )

        self.discriminator = torch.nn.Sequential(
            Flatten(),
            MLP(self.input_size, 1, discriminator_features, 
                discriminator_hidden_layers, lambda x: F.leaky_relu(x, 0.01)),
            torch.nn.Sigmoid(),
        )

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
        real_p = self.discriminate(real)
        fake_p = self.discriminate(fake)

        discriminator_loss = - torch.log(real_p + 1e-8) - torch.log(1 - fake_p + 1e-8)

        return torch.mean(discriminator_loss)

    def get_generator_loss(self, fake):
        fake_p = self.discriminate(fake)

        generator_loss = - torch.log(fake_p + 1e-8)

        return torch.mean(generator_loss)