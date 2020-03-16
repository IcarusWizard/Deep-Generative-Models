import torch
import numpy as np
from torch import nn

from ..modules import MLP, Flatten, Unflatten, ResNet, ResBlock
from functools import partial

class MLPEncoder(torch.nn.Module):
    def __init__(self, c, h, w, latent_dim, features, hidden_layers):
        super().__init__()
        input_dim = h * w * c
        self.encoder = torch.nn.Sequential(
            Flatten(),
            MLP(input_dim, 2 * latent_dim, features, hidden_layers),
        )

    def forward(self, x):
        return self.encoder(x)

class MLPDecoder(torch.nn.Module):
    def __init__(self, output_c, h, w, latent_dim, features, hidden_layers):
        super().__init__()
        output_dim = output_c * h * w
        self.decoder = torch.nn.Sequential(
            MLP(latent_dim, output_dim, features, hidden_layers),
            Unflatten([output_c, h, w]),
        )
    
    def forward(self, x):
        return self.decoder(x)

class ConvEncoder(torch.nn.Module):
    def __init__(self, c, h, w, latent_dim, conv_features, down_sampling, res_layers, mlp_features, mlp_layers, batchnorm):
        super().__init__()

        feature_shape = (conv_features, h // (2 ** down_sampling), w // (2 ** down_sampling))
        
        conv_features = conv_features // (2 ** down_sampling)
        encoder_list = [
            torch.nn.Conv2d(c, conv_features, 7, 1, padding=3),
            torch.nn.ReLU(inplace=True)
        ]

        for i in range(down_sampling):
            encoder_list.append(torch.nn.Conv2d(conv_features, conv_features * 2, 3, 2, padding=1))
            encoder_list.append(torch.nn.ReLU(inplace=True))
            conv_features *= 2
            for j in range(res_layers[i]):
                encoder_list.append(ResBlock(conv_features, batchnorm))

        encoder_list.append(Flatten())
        encoder_list.append(MLP(np.prod(feature_shape), 2 * latent_dim, mlp_features, mlp_layers, partial(torch.nn.LeakyReLU, negative_slope=0.1)))

        self.encoder = torch.nn.Sequential(*encoder_list)

    def forward(self, x):
        return self.encoder(x)

class ConvDecoder(torch.nn.Module):
    def __init__(self, output_c, h, w, latent_dim, conv_features, down_sampling, res_layers, mlp_features, mlp_layers, batchnorm):
        super().__init__()
        res_layers = list(reversed(res_layers))
        
        feature_shape = (conv_features, h // (2 ** down_sampling), w // (2 ** down_sampling))

        decoder_list = [
            MLP(latent_dim, np.prod(feature_shape), mlp_features, mlp_layers, partial(torch.nn.LeakyReLU, negative_slope=0.1)),
            Unflatten(feature_shape),
        ]

        conv_features = conv_features
        for i in range(down_sampling):
            for j in range(res_layers[i]):
                decoder_list.append(ResBlock(conv_features, batchnorm))
            decoder_list.append(torch.nn.ConvTranspose2d(conv_features, conv_features // 2, 4, 2, padding=1))
            decoder_list.append(torch.nn.ReLU(inplace=True))
            conv_features //= 2
        
        decoder_list.append(torch.nn.Conv2d(conv_features, output_c, 3, stride=1, padding=1))
        
        self.decoder = torch.nn.Sequential(*decoder_list)

    def forward(self, x):
        return self.decoder(x)

class FullConvEncoder(torch.nn.Module):
    def __init__(self, input_c, h, w, output_c, conv_features, down_sampling, res_layers, batchnorm):
        super().__init__()
        
        conv_features = conv_features // (2 ** down_sampling)
        encoder_list = [
            torch.nn.Conv2d(input_c, conv_features, 7, 1, padding=3),
            torch.nn.ReLU(inplace=True)
        ]

        for i in range(down_sampling):
            encoder_list.append(torch.nn.Conv2d(conv_features, conv_features * 2, 3, 2, padding=1))
            encoder_list.append(torch.nn.ReLU(inplace=True))
            conv_features *= 2
            for j in range(res_layers[i]):
                encoder_list.append(ResBlock(conv_features, batchnorm))

        encoder_list.append(torch.nn.Conv2d(conv_features, output_c, 3, stride=1, padding=1))

        self.encoder = torch.nn.Sequential(*encoder_list)

    def forward(self, x):
        return self.encoder(x)

class FullConvDecoder(torch.nn.Module):
    def __init__(self, input_c, h, w, output_c, conv_features, down_sampling, res_layers, batchnorm):
        super().__init__()
        res_layers = list(reversed(res_layers))

        decoder_list = [
            torch.nn.Conv2d(input_c, conv_features, 3, stride=1, padding=1)
        ]

        conv_features = conv_features
        for i in range(down_sampling):
            for j in range(res_layers[i]):
                decoder_list.append(ResBlock(conv_features, batchnorm))
            decoder_list.append(torch.nn.ConvTranspose2d(conv_features, conv_features // 2, 4, 2, padding=1))
            decoder_list.append(torch.nn.ReLU(inplace=True))
            conv_features //= 2
        
        decoder_list.append(torch.nn.Conv2d(conv_features, output_c, 3, stride=1, padding=1))
        
        self.decoder = torch.nn.Sequential(*decoder_list)

    def forward(self, x):
        return self.decoder(x)

class NearestEmbed(torch.nn.Module):
    r"""
        Embedding Module for VQ-VAE
        
        Inputs:

            k : int, size of the embedding space (number of the vector)
            d : int, dimension of each embedding vector
    """
    def __init__(self, k, d):
        super(NearestEmbed, self).__init__()
        self.k = k
        self.d = d
        self.weight = torch.nn.Parameter(torch.rand(d, k))
        self.weight.data.uniform_(-1./k, 1./k)

    def forward(self, x):
        r"""
            Inputs:

                x : tensor[batch_size, d, *] (* can be arbitrary)]

            Outputs:

                quantized_x : tensor[batch_size, d, *]
                index : tensor[batch_size, d], the index of selected embedding vectors 
        """
        batch_size = x.shape[0]
        num_latents = int(np.prod(x.shape[2:]))
        dims = list(range(len(x.size())))

        # expand so it broadcastable
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        if num_arbitrary_dims:
            emb_expanded = self.weight.view(self.d, *([1] * num_arbitrary_dims), self.k)
        else:
            emb_expanded = self.weight

        # find nearest neighbors
        dist = torch.sum((x_expanded - emb_expanded) ** 2, dim=1)
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        quantized = self.weight.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])

        return quantized.contiguous(), argmin

    def select(self, index):
        r"""
            Select the embedding vectors by index

            Inputs:

                index : tensor[*]

            Outputs:

                selected_vectors : tensor[*, d] 
                
        """
        shape = index.shape
        return self.weight.t().index_select(0, index.view(-1)).view(*shape, self.d)