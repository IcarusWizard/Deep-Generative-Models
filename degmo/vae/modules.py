import torch
import numpy as np

from ..modules import MLP, Flatten, Unflatten, ResNet, ResBlock

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
        encoder_list.append(MLP(np.prod(feature_shape), 2 * latent_dim, mlp_features, mlp_layers))

        self.encoder = torch.nn.Sequential(*encoder_list)

    def forward(self, x):
        return self.encoder(x)

class ConvDecoder(torch.nn.Module):
    def __init__(self, output_c, h, w, latent_dim, conv_features, down_sampling, res_layers, mlp_features, mlp_layers, batchnorm):
        super().__init__()
        res_layers = list(reversed(res_layers))
        
        feature_shape = (conv_features, h // (2 ** down_sampling), w // (2 ** down_sampling))

        decoder_list = [
            MLP(latent_dim, np.prod(feature_shape), mlp_features, mlp_layers),
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