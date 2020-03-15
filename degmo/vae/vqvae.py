import torch 
import torchvision
from torch.functional import F
import numpy as np
from torch import nn

from .modules import FullConvEncoder, FullConvDecoder, NearestEmbed
from .utils import LOG2PI

class VQ_VAE(torch.nn.Module):
    r"""
        VQ-VAE: http://arxiv.org/abs/1711.00937
        
        Inputs:

            c : int, channel of the input image
            h : int, height of the input image
            w : int, width of the input image
            k : int, size of the embedding space (number of the vector)
            d : int, dimension of each embedding vector
            network_type : str, type of the encoder and decoder, choose from fullcomv, conv and mlp, default: fullconv
            config : dict, parameters for constructe encoder and decoder
            output_type : str, type of the distribution p(x|z), choose from fix_std(std=1), gauss and bernoulli, default: gauss
            beta : int, coefficient of the commitment loss
    """
    def __init__(self, c=3, h=32, w=32, k=512, d=64, network_type='fullconv', config={},
                 output_type='gauss', beta=0.25):    
        super().__init__()
        self.k = k
        self.d = d
        self.network_type = network_type
        self.output_type = output_type
        self.beta = beta
        output_c = 2 * c if self.output_type == 'gauss' else c

        self.encoder = FullConvEncoder(c, h, w, d, **config)

        with torch.no_grad():
            sample_data = torch.randn(1, c, h, w)
            sample_output = self.encoder(sample_data)
            self.latent_shape = sample_output.shape[2:]

        self.embedding = NearestEmbed(k, d)

        self.decoder = FullConvDecoder(d, h, w, output_c, **config)

    def forward(self, x):
        batch_size = x.shape[0]
        z_e = self.encoder(x) # encoder to latent space

        z_q, argmin = self.embedding(z_e, weight_sg=True) # find the nearest embedding

        emb, _ = self.embedding(z_e.detach()) # use to compute gradient

        vq_loss = torch.mean(torch.sum(torch.norm((emb - z_e.detach()), 2, 1), dim=(1, 2)))
        commit_loss = torch.mean(torch.sum(torch.norm((emb.detach() - z_e), 2, 1), dim=(1, 2)))

        if self.output_type == 'fix_std':
            _x = self.decoder(z_q)
            reconstruction_loss = torch.mean(torch.sum((x - _x) ** 2 / 2 + LOG2PI, dim=(1, 2, 3)))

        elif self.output_type == 'gauss':
            output_mu, logs = torch.chunk(self.decoder(z_q), 2, 1)
            logs = torch.tanh(logs)
            reconstruction_loss = torch.mean(torch.mean((x - output_mu) ** 2 / 2 * torch.exp(-2 * logs) + LOG2PI + logs, dim=(1, 2, 3)))

        info = {
            "resonstraction_loss" : reconstruction_loss, 
            "vq_loss" : vq_loss, 
            "commitment_loss" :commit_loss
        }

        return reconstruction_loss + vq_loss + self.beta * commit_loss, info

    def encode(self, x):
        z_e = self.encoder(x) # encoder to latent space

        z_q, argmin = self.embedding(z_e, weight_sg=True)

        return z_q
    
    def decode(self, z, deterministic=True):
        if self.output_type == 'fix_std':
            output_mu = self.decoder(z)
            result = output_mu if deterministic else output_mu + torch.randn_like(output_mu)
        elif self.output_type == 'gauss':
            output_mu, logs = torch.chunk(self.decoder(z), 2, 1)
            logs = torch.tanh(logs)
            result = output_mu if deterministic else output_mu + torch.randn_like(logs) * torch.exp(logs)
        return result

    def sample(self, num, deterministic=True):
        index = torch.randint(self.k, size=(num, *self.latent_shape))
        z = self.embedding.select(index).permute(0, 3, 1, 2)
        return self.decode(z, deterministic=deterministic)