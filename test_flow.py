import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

import os, argparse

from models.utils import step_loader, select_gpus, config_dataset
from models.flow.test_utils import config_model, generation, extrapolation, interpolation

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

    # config model
    model, checkpoint = config_model(train_time_args, args, checkpoint)

    # load state dict
    state_dict = checkpoint['model_state_dict']
    model_state_dick = model.state_dict()
    for k in list(state_dict.keys()):
        if 'mask' in k: state_dict[k] = model_state_dick[k]
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    model = model.to(device)

    # run test
    if args.mode == 'generation':
        generation(model, args.temperature)

    elif args.mode == 'extrapolation': 
        extrapolation(model, args.temperature, args.scale)

    elif args.mode == 'interpolation':
        _, _, train_loader, _, _ = config_dataset(train_time_args, 2)
        interpolation(model, train_loader)

    else:
        raise ValueError('Mode {} is not supported!'.format(args.mode))
