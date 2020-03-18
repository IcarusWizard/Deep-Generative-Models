import torch
from torch.functional import F

from .modules import MLPGenerator, MLPDiscriminator
from .trainer import AdversarialTrainer

class GAN(torch.nn.Module):
    def __init__(self, c, h, w, latent_dim, 
        discriminator_features, discriminator_hidden_layers,
        generator_features, generator_hidden_layers):
        super().__init__()

        self.input_size = c * h * w
        self.latent_dim = latent_dim

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.generator = MLPGenerator(c, h, w, latent_dim, generator_features, generator_hidden_layers)

        self.discriminator = MLPDiscriminator(c, h, w, discriminator_features, discriminator_hidden_layers)

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

    def get_trainer(self):
        return AdversarialTrainer