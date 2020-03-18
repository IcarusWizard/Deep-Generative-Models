import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

import os, argparse

import degmo
from degmo.utils import step_loader, select_gpus, config_dataset
from degmo.flow.run_utils import config_model_test, generation, extrapolation, interpolation
from degmo import LOGDIR, MODELDIR, VERSION, CONFIG_PATH

CONFIG = os.path.join(CONFIG_PATH, 'flow.json')
LOGDIR = os.path.join(LOGDIR, 'flow')
MODELDIR = os.path.join(MODELDIR, 'flow')

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

    checkpoint = torch.load(os.path.join(MODELDIR, args.model + '.pt'), map_location='cpu')

    config = checkpoint['config']
    print('load model: {}'.format(config['model']))
    print('seed: {}'.format(checkpoint['seed']))
    print('model parameters: {}'.format(checkpoint['model_parameters']))

    # restore checkpoint
    model = config_model_test(args, checkpoint)
    model = model.to(device)

    # run test
    if args.mode == 'generation':
        generation(model, args.temperature)

    elif args.mode == 'extrapolation': 
        extrapolation(model, args.temperature, args.scale)

    elif args.mode == 'interpolation':
        _, _, train_loader, _, _ = config_dataset(config, 2)
        interpolation(model, train_loader)

    else:
        raise ValueError('Mode {} is not supported!'.format(args.mode))
