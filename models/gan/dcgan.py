import torch
from torch.functional import F
import numpy as np

from ..modules import Flatten, Unflatten
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

        self.criterion = torch.nn.BCELoss()

        generator = []
        out_features = generator_features

        generator.append(torch.nn.Linear(latent_dim, out_features * self.out_h * self.out_w, bias=False))
        generator.append(Unflatten(out_features, self.out_h, self.out_w))
        
        for i in range(downsampling - 1):
            generator.append(torch.nn.BatchNorm2d(out_features))
            generator.append(torch.nn.ReLU(True))
            if i == 0 and h == 28: # mnist
                generator.append(torch.nn.ConvTranspose2d(out_features, out_features // 2, 4, 
                    stride=2, padding=2, output_padding=1, bias=False))
            else:
                generator.append(torch.nn.ConvTranspose2d(out_features, out_features // 2, 4, stride=2, padding=1, bias=False))

            out_features = out_features // 2

        generator.append(torch.nn.BatchNorm2d(out_features))
        generator.append(torch.nn.ReLU(True))
        generator.append(torch.nn.ConvTranspose2d(out_features, c, 4, stride=2, padding=1, bias=False))
        generator.append(torch.nn.Tanh())

        self.generator = torch.nn.Sequential(*generator)

        discriminator = []
        in_features = c
        out_features = discriminator_features // (2 ** (downsampling - 1))
        
        for i in range(downsampling):
            if i == downsampling - 1 and h == 28: # mnist
                discriminator.append(torch.nn.Conv2d(in_features, out_features, 4, stride=2, padding=2, bias=False))
            else:
                discriminator.append(torch.nn.Conv2d(in_features, out_features, 4, stride=2, padding=1, bias=False))
            discriminator.append(torch.nn.BatchNorm2d(out_features))
            discriminator.append(torch.nn.LeakyReLU(0.2, True))

            in_features = out_features
            out_features *= 2

        discriminator.append(Flatten())
        discriminator.append(torch.nn.Linear(in_features * self.out_h * self.out_w, 1, bias=False))
        discriminator.append(torch.nn.Sigmoid())

        self.discriminator = torch.nn.Sequential(*discriminator)
        print(self.discriminator)
        print(self.generator)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

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
        device = real.device

        real_p = self.discriminate(real)
        fake_p = self.discriminate(fake)

        fake_label = torch.zeros(*fake_p.shape, device=device)
        real_label = torch.ones(*real_p.shape, device=device)

        discriminator_loss = self.criterion(real_p, real_label) + self.criterion(fake_p, fake_label)

        return torch.mean(discriminator_loss)

    def get_generator_loss(self, fake):
        device = fake.device

        fake_p = self.discriminate(fake)

        real_label = torch.ones(*fake_p.shape, device=device)
        generator_loss = self.criterion(fake_p, real_label)

        return torch.mean(generator_loss)