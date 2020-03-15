import torch
import numpy as np
from torch import nn

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

class NearestEmbedFunc(torch.autograd.Function):
    r"""
        Gradient function that perform nearest embedding

        Input:

            x : tensor[batch_size, d, *] (* can be arbitrary)
            emb : tensor[k, d], embedding vectors
        """
    @staticmethod
    def forward(ctx, x, emb):
        if x.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: x.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(x.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = x.size(0)
        ctx.num_latents = int(np.prod(x.shape[2:]))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(x)
        ctx.dims = list(range(len(x.size())))

        # expand so it broadcastable
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, dim=1)
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin) 
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None

        if ctx.needs_input_grad[0]:
            # copy the output gradient for the input
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            # accumulate the gradient for every selected embedding vector
            argmin, = ctx.saved_tensors
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            # idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output.data)
            idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output)
            n_idx_choice = idx_choices.sum(0) # the times of e_d choosed in this minibatch
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice # weight of every choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            # grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) * idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
            grad_emb = torch.sum(grad_output.unsqueeze(-1) * idx_avg_choices.unsqueeze(1), 0)

        return (grad_input, grad_emb, )

def nearest_embed(x, emb):
    return NearestEmbedFunc.apply(x, emb)

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

    def forward(self, x, weight_sg=False):
        r"""
            Inputs:

                x : tensor[batch_size, d, *] (* can be arbitrary)
                weight_sg : bool, whether the operation create a gradient for embedding vectors
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)

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
        