import torch 
import torchvision
from torch.functional import F
import numpy as np
from torch import nn

from degmo.autoregressive.pixelcnn import PixelCNN
from .modules import FullConvEncoder, FullConvDecoder, ConvEncoder, ConvDecoder, MLPEncoder, MLPDecoder, \
    Flatten, Unflatten, NearestEmbed
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

        self.embedding = NearestEmbed(k, d)

        if network_type == 'fullconv':
            self.encoder = FullConvEncoder(c, h, w, d, **config)
            self.decoder = FullConvDecoder(d, h, w, output_c, **config)
        else:
            latent_dim = config.pop('latent_dim')
            if network_type == 'conv':
                self.encoder = torch.nn.Sequential(
                    ConvEncoder(c, h, w, latent_dim * d // 2, **config),
                    Unflatten([d, latent_dim])
                )
                self.decoder = torch.nn.Sequential(
                    Flatten(),
                    ConvDecoder(output_c, h, w, latent_dim * d, **config)
                )
            else:
                self.encoder = torch.nn.Sequential(
                    MLPEncoder(c, h, w, latent_dim * d // 2, **config),
                    Unflatten([d, latent_dim])
                )
                self.decoder = torch.nn.Sequential(
                    Flatten(),
                    MLPDecoder(output_c, h, w, latent_dim * d, **config)
                )                

        with torch.no_grad():
            sample_data = torch.randn(1, c, h, w)
            sample_output = self.encoder(sample_data)
            self.latent_shape = sample_output.shape[2:]

        if self.network_type == 'fullconv':
            self.prior = PixelCNN(1, self.latent_shape[0], self.latent_shape[1], mode='res', first_kernel_size=3,
                                  features=64, layers=7, post_features=1024, bits=np.log2(self.k))
        else:
            # TODO: Implement prior models for flatten latent space
            self.prior = None


    def forward(self, x):
        batch_size = x.shape[0]
        z_e = self.encoder(x) # encoder to latent space

        z_q, index = self.embedding(z_e) # find the nearest embedding

        vq_loss = torch.sum((z_q - z_e.detach()) ** 2) / batch_size
        commit_loss = torch.sum((z_q.detach() - z_e) ** 2) / batch_size

        z_q = z_e + (z_q - z_e).detach() # this trick provides gradient to the encoder

        if self.output_type == 'fix_std':
            _x = self.decoder(z_q)
            reconstruction_loss = torch.sum((x - _x) ** 2 / 2 + LOG2PI) / batch_size

        elif self.output_type == 'gauss':
            output_mu, logs = torch.chunk(self.decoder(z_q), 2, 1)
            logs = torch.tanh(logs)
            reconstruction_loss = torch.sum((x - output_mu) ** 2 / 2 * torch.exp(-2 * logs) + LOG2PI + logs) / batch_size

        loss = reconstruction_loss + vq_loss + self.beta * commit_loss
        info = {
            "resonstraction_loss" : reconstruction_loss, 
            "vq_loss" : vq_loss, 
            "commitment_loss" :commit_loss
        }

        if self.prior:
            if self.network_type == 'fullconv':
                prior_input = (index.detach().type_as(x) / (self.k - 1)).unsqueeze(1)
                prior_loss = self.prior(prior_input)
                loss += prior_loss
                info['prior_loss'] = prior_loss 
            else:
                pass

        return loss, info

    def encode(self, x):
        z_e = self.encoder(x) # encoder to latent space

        z_q, index = self.embedding(z_e)

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
        if self.prior:
            index = (self.prior.sample(num).squeeze(1) * (self.k - 1)).long()
        else:
            index = torch.randint(self.k, size=(num, *self.latent_shape)).to(self.embedding.weight.device)
        dims = list(range(len(index.shape) + 1))
        z = self.embedding.select(index).permute(0, dims[-1], *dims[1:-1]).contiguous()
        return self.decode(z, deterministic=deterministic)