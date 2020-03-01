import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter

import os, argparse
from tqdm import tqdm

from degmo.gan.test_utils import config_model, generation, manifold, interpolation, helix_interpolation
from degmo.utils import setup_seed, select_gpus, nats2bits, config_dataset, load_config
from degmo import LOGDIR, MODELDIR, VERSION, CONFIG_PATH

CONFIG = os.path.join(CONFIG_PATH, 'gan.json')
LOGDIR = os.path.join(LOGDIR, 'gan')
MODELDIR = os.path.join(MODELDIR, 'gan')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='file name of the stored model')
    parser.add_argument('--mode', type=str, default='generation',
                        help='test mode, select from generation, linear_manifold, circular_manifold, interpolation')
    parser.add_argument('--truncation', type=float, default=1.0,
                        help='truncation trick for generation')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    
    # config gpu
    select_gpus(args.gpu) 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(MODELDIR + args.model + '.pt', map_location='cpu')

    if checkpoint['version'] != VERSION:
        print('Warning: model version {} doesn\'t match lib version {}!'.format(checkpoint['version'], VERSION))

    train_time_args = checkpoint['train_time_args']
    print('load model: {}'.format(train_time_args.model))
    print('seed: {}'.format(checkpoint['seed']))
    print('model parameters: {}'.format(checkpoint['model_parameters']))

    # config model
    generator, latent_dim = config_model(train_time_args, checkpoint)
    generator = generator.to(device)

    if args.mode == 'generation':
        generation(generator, latent_dim, args.truncation)
    elif args.mode == 'linear_manifold':
        manifold(generator, latent_dim, mode='linear')
    elif args.mode == 'circular_manifold':
        manifold(generator, latent_dim, mode='circular')
    elif args.mode == 'interpolation':
        writer = SummaryWriter(os.path.join(LOGDIR, 'GIF', args.model))
        interpolation(generator, latent_dim, writer, args.truncation)
    elif args.mode == 'helix_interpolation':
        writer = SummaryWriter(os.path.join(LOGDIR, 'GIF', args.model))
        helix_interpolation(generator, latent_dim, writer, args.truncation)        
    else:
        raise ValueError('Mode {} is not supported!'.format(args.mode))

    # # open log
    # writer = SummaryWriter(LOGDIR + '{}'.format(args.model))