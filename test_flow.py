import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

import os, argparse

from models.flow import NICE, RealNVP2D, GLOW
from models.utils import step_loader, select_gpus, config_dataset

from models.flow import LOGDIR, MODELDIR, VERSION

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='file name of the stored model')
    parser.add_argument('--mode', type=str, default='generation',
                        help='test mode, select from generation, interpolation, extrapolation')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature used for generation')
    parser.add_argument('--scale', type=int, default=2,
                        help='times of scale multiplication, choose from 2, 4, 8')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    select_gpus(args.gpu) # config gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(MODELDIR + args.model + '.pt', map_location='cpu')

    if checkpoint['version'] != VERSION:
        print('Warning: model version {} doesn\'t match lib version {}!'.format(checkpoint['version'], VERSION))

    train_time_args = checkpoint['train_time_args']
    print('load model: {}'.format(train_time_args.model))
    print('seed: {}'.format(checkpoint['seed']))
    print('model parameters: {}'.format(checkpoint['model_parameters']))

    _, _, train_loader, _, _ = config_dataset(train_time_args, 2)
    
    if train_time_args.model == 'NICE':
        assert not args.mode == 'extrapolation', 'NICE do not support mode extrapolation'
        model = NICE(**checkpoint['model_parameters'])
    elif train_time_args.model == 'RealNVP':
        checkpoint['model_parameters']['h'] *= args.scale
        checkpoint['model_parameters']['w'] *= args.scale
        model = RealNVP2D(**checkpoint['model_parameters'])
    elif train_time_args.model == 'Glow':
        checkpoint['model_parameters']['h'] *= args.scale
        checkpoint['model_parameters']['w'] *= args.scale
        model = GLOW(**checkpoint['model_parameters'])
    else:
        raise ValueError('Model {} is not supported!'.format(train_time_args.model))

    state_dict = checkpoint['model_state_dict']
    model_state_dick = model.state_dict()
    for k in list(state_dict.keys()):
        if 'mask' in k: state_dict[k] = model_state_dick[k]
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    model = model.to(device)

    if args.mode == 'generation':
        with torch.no_grad():
            imgs = torch.clamp(model.sample(64, temperature=args.temperature), 0, 1)
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

    elif args.mode == 'extrapolation':
        size = 8 // args.scale
        with torch.no_grad():
            imgs = torch.clamp(model.sample(size ** 2, temperature=args.temperature), 0, 1)
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

    elif args.mode == 'interpolation':
        step = 0
        imgs = []
        with torch.no_grad():
            for batch in step_loader(train_loader):
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

    else:
        raise ValueError('Mode {} is not supported!'.format(args.mode))
