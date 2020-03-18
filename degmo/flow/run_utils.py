import torch
import numpy as np
import matplotlib.pyplot as plt

from . import *

def config_model_train(config, model_param):
    model_param.update({
        "features" : config['features'],
        "bits" : config['bits'],
    })
    if config['model']== 'NICE':
        model_param.update({
            "hidden_layers" : config['blocks'],
        })
        model = NICE(**model_param)        
    elif config['model']== 'RealNVP':
        model_param.update({
            "hidden_blocks" : config['blocks'],
            "down_sampling" : config['down_sampling'],
        })
        model = RealNVP2D(**model_param)
    elif config['model']== 'Glow':
        model_param.update({
            "K" : config['K'],
            "L" : config['L'],
            "use_lu" : not config['nolu'],
            "coupling" : config['coupling'],
        })
        model = GLOW(**model_param)
    else:
        raise ValueError('Model {} is not supported!'.format(config['model']))

    return model, model_param

def config_model_test(args, checkpoint):
    config = checkpoint['config']
    # build model
    if config['model'] == 'NICE':
        assert not args.mode == 'extrapolation', 'NICE do not support mode extrapolation'
        model = NICE(**checkpoint['model_parameters'])
    elif config['model'] == 'RealNVP':
        if args.mode == 'extrapolation':
            checkpoint['model_parameters']['h'] *= args.scale
            checkpoint['model_parameters']['w'] *= args.scale
        model = RealNVP2D(**checkpoint['model_parameters'])
    elif config['model'] == 'Glow':
        if args.mode == 'extrapolation':
            checkpoint['model_parameters']['h'] *= args.scale
            checkpoint['model_parameters']['w'] *= args.scale
        model = GLOW(**checkpoint['model_parameters'])
    else:
        raise ValueError('Model {} is not supported!'.format(config['model']))

    # load state dict
    state_dict = checkpoint['model_state_dict']
    model_state_dick = model.state_dict()
    for k in list(state_dict.keys()):
        if 'mask' in k: state_dict[k] = model_state_dick[k] # replace mask matrix when running extrapolation
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def generation(model, temperature):
    """
        Generate 64 samples from learned model
    """
    with torch.no_grad():
        imgs = torch.clamp(model.sample(64, temperature=temperature), 0, 1)
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

def extrapolation(model, temperature, scale):
    """
        Extrapolation for learned model, only support scale equal to 2, 4, 8
        See more information in Appendix C in RealNVP paper
    """
    size = 8 // scale
    with torch.no_grad():
        imgs = torch.clamp(model.sample(size ** 2, temperature=temperature), 0, 1)
        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()

    if imgs.shape[3] == 1:
        imgs = imgs[:, :, :, 0]
    fig, axes = plt.subplots(size, size, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    if size == 1:
        axes.imshow(imgs[0])
        axes.axis('off')
    else:            
        for i in range(size):
            for j in range(size):
                axes[i, j].imshow(imgs[i*size+j])
                axes[i, j].axis('off')
    plt.show()    

def interpolation(model, loader):
    """
        Linear interpolation between pair of training data in the latent space.
    """
    device = next(model.parameters()).device
    step = 0
    imgs = []
    with torch.no_grad():
        for batch in iter(loader):
            batch = batch[0].to(device)

            z, _ = model(batch)

            z1, z2 = z[0], z[1]

            delta = (z2 - z1) / 9.0

            zs = [(z1 + i * delta).unsqueeze(dim=0) for i in range(10)]
            zs = torch.cat(zs, dim=0)

            xs = torch.clamp(model.z2x(zs), 0, 1)

            imgs.append(xs.cpu().permute(0, 2, 3, 1).numpy())

            step = step + 1

            if step >= 10:
                break

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    for i, xs in enumerate(imgs):
        if xs.shape[3] == 1:
            xs = xs[:, :, :, 0]
        for j in range(10):
            axes[i, j].imshow(xs[j])
            axes[i, j].axis('off')
    plt.show()