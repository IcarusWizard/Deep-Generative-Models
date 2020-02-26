import torch
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

import os, tempfile
from moviepy import editor as mpy

from . import GAN, DCGAN, WGAN, WGANGP, SNGAN, SAGAN

def config_model(train_time_args, checkpoint):
    # build model
    if train_time_args.model == 'GAN':
        model = GAN(**checkpoint['model_parameters'])
    elif train_time_args.model == 'DCGAN':
        model = DCGAN(**checkpoint['model_parameters'])
    elif train_time_args.model == 'WGAN':
        model = WGAN(**checkpoint['model_parameters'])
    elif train_time_args.model == 'WGAN-GP':
        model = WGANGP(**checkpoint['model_parameters'])
    elif train_time_args.model == 'SNGAN':
        model = SNGAN(**checkpoint['model_parameters'])
    elif train_time_args.model == 'SAGAN':
        model = SAGAN(**checkpoint['model_parameters'])

    # only restore generator checkpoint
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.generator.eval()

    latent_dim = model.latent_dim

    return model.generator, latent_dim

def generation(generator, latent_dim, truncation):
    """
        Generate 64 samples from learned model
    """
    device = next(generator.parameters()).device
    dtype = next(generator.parameters()).dtype
    with torch.no_grad():
        z = truncnorm.rvs(-2, 2, size=(64, latent_dim)) * truncation
        z = torch.tensor(z, dtype=dtype, device=device)
        imgs = torch.clamp(generator(z) / 2 + 0.5, 0, 1)
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()

    if imgs.shape[3] == 1:
        imgs = imgs[:, :, :, 0]
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(8):
        for j in range(8):
            axes[i, j].imshow(imgs[i*8+j])
            axes[i, j].axis('off')
    plt.show()

def length(vector):
    return np.sqrt(np.sum(vector ** 2))

def generate_plane(space_size):
    """
        Random generation of a plane in R^(space_size),
        return a pair of basis
    """
    x, y = np.random.randn(space_size), np.random.randn(space_size)

    # orthogonalization
    y = y - np.sum(x * y) / length(x) ** 2 * x

    # normalization
    x = x / length(x)
    y = y / length(y)

    return x, y

def manifold(generator, latent_dim, mode='linear', num=9):
    """
        visulize a learned manifold
    """
    basis_x, basis_y = generate_plane(latent_dim)

    scale = latent_dim ** 0.5

    if mode == 'linear':
        x = np.linspace(-scale, scale, num=9)
        grid_x, grid_y = np.meshgrid(x, x)
    elif mode == 'circular':
        assert num % 2 == 1, "only support odd number"
        grid_x = np.zeros((num, num))
        grid_y = np.zeros((num, num))

        def generate_pos(level):
            # generate circular position counterclockwise
            center = num // 2

            for i in range(0, level + 1):
                yield (center - i, center + level)

            for i in range(-level + 1, level + 1):
                yield (center - level, center - i)

            for i in range(-level + 1, level + 1):
                yield (center + i, center - level)

            for i in range(-level + 1, level + 1):
                yield (center + level, center + i)

            for i in range(-level + 1, 0):
                yield (center - i, center + level)

        for level in range(1, num // 2 + 1):
            radius = level * scale / (num // 2)
            _theta = np.linspace(0, 2 * np.pi, 8 * level)

            for i, pos in enumerate(generate_pos(level)):
                grid_x[pos] = radius * np.cos(_theta[i])
                grid_y[pos] = radius * np.sin(_theta[i])

    z = grid_x.reshape((num, num, 1)) * basis_x.reshape((1, 1, latent_dim)) + \
        grid_y.reshape((num, num, 1)) * basis_y.reshape((1, 1, latent_dim))

    z = z.reshape((-1, latent_dim))

    device = next(generator.parameters()).device
    dtype = next(generator.parameters()).dtype

    with torch.no_grad():
        z = torch.tensor(z, dtype=dtype, device=device)
        imgs = torch.clamp(generator(z) / 2 + 0.5, 0, 1)
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()

    if imgs.shape[3] == 1:
        imgs = imgs[:, :, :, 0]
    fig, axes = plt.subplots(num, num, figsize=(num, num))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i in range(num):
        for j in range(num):
            axes[i, j].imshow(imgs[i*num+j])
            axes[i, j].axis('off')
    plt.show()    

def points_linear_interpolation(z1, z2, frames, focus=False):
    """
        linear interpolation from tensor z1 to z2, 
        the result does not contain endpoint 
    """
    eps = np.linspace(0, 1, num=frames, endpoint=False)
    if focus:
        eps = 0.5 - 0.5 * np.cos(np.pi * eps) # make the interpolation more focus on position near z1 and z2

    return torch.cat([(z1 * (1 - _eps) + z2 * _eps).unsqueeze(dim=0) for _eps in eps], dim=0)
    
def interpolation(generator, latent_dim, writer, truncation, num=10, frames=30):
    """
        generate linear interpolation video for the given generator
    """
    device = next(generator.parameters()).device
    dtype = next(generator.parameters()).dtype
    
    imgs = []
    with torch.no_grad():
        z = truncnorm.rvs(-2, 2, size=(num, latent_dim)) * truncation
        z = torch.tensor(z, dtype=dtype, device=device)

        for i in range(num):
            _z = points_linear_interpolation(z[i], z[(i + 1) % num], frames)
            _imgs = torch.clamp(generator(_z) / 2 + 0.5, 0, 1)
            imgs.append(_imgs.unsqueeze(dim=0))
    
        imgs = torch.cat(imgs, dim=1).cpu() # 1 x T x C x H x W

    # save to tensorboard 
    writer.add_video('interpolation', imgs, fps=frames)

    # save gif to temp folder
    filename = os.path.join(tempfile.gettempdir(), 'interpolation.gif')
    imgs = imgs[0].permute(0, 2, 3, 1).numpy() * 255
    clip = mpy.ImageSequenceClip([imgs[i] for i in range(imgs.shape[0])], fps=frames)
    clip.write_gif(filename, verbose=False, logger=None)
    print('gif has written to {}'.format(filename))

def helix_interpolation(generator, latent_dim, writer, truncation, num=10, frames=120):
    """
        generate helix interpolation video for the given generator
    """
    device = next(generator.parameters()).device
    dtype = next(generator.parameters()).dtype
    
    imgs = []
    with torch.no_grad():
        # generate Archimedean Spiral
        max_radius = latent_dim ** 0.5

        basis_x, basis_y = generate_plane(latent_dim)

        total_frames = num * frames
        t = np.arange(total_frames)
        theta = (2 * np.pi * t / frames).reshape((total_frames, 1))
        radius = (t / total_frames * max_radius).reshape((total_frames, 1))

        z = radius * np.cos(theta) * basis_x.reshape((1, latent_dim)) + \
            radius * np.sin(theta) * basis_y.reshape((1, latent_dim))

        z = torch.tensor(z, dtype=dtype, device=device)

        imgs = []
        for _z in torch.chunk(z, 10, dim=0):
            _imgs = torch.clamp(generator(_z) / 2 + 0.5, 0, 1).cpu()
            imgs.append(_imgs)
        imgs = torch.cat(imgs, dim=0).cpu() # T x C x H x W

    # save to tensorboard 
    writer.add_video('interpolation', imgs.unsqueeze(dim=0), fps=frames)

    # save gif to temp folder
    filename = os.path.join(tempfile.gettempdir(), 'interpolation.gif')
    imgs = imgs.permute(0, 2, 3, 1).numpy() * 255
    clip = mpy.ImageSequenceClip([imgs[i] for i in range(imgs.shape[0])], fps=frames // 4)
    clip.write_gif(filename, verbose=False, logger=None)
    print('gif has written to {}'.format(filename))