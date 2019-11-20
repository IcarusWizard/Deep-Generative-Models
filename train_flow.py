import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter

import os, argparse

from models.flow import config_model, train_flow
from models.utils import config_dataset, setup_seed, step_loader, select_gpus, nats2bits, load_config

from models.flow import LOGDIR, MODELDIR, VERSION, CONFIG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='choose from bmnist, mnist, svhn, cifar, celeba32, celeba64, celeba128')
    parser.add_argument('--model', type=str, default='NICE',
                        help='choose from NICE, RealNVP, Glow')
    parser.add_argument('--custom', action='store_true', help='enable custom config')

    model_parser = parser.add_argument_group('model', 'parameters for model config')
    model_parser.add_argument('--blocks', type=int, default=8)
    model_parser.add_argument('--features', type=int, default=256)
    model_parser.add_argument('--bits', type=int, default=8, help='bits per dimension in the dataset')
    model_parser.add_argument('--coupling', type=str, default='affine', help='choose from affine and additive')
    model_parser.add_argument('--actnorm_batch_size', type=int, default=128, help='batch size for initializing ActNorm')
    model_parser.add_argument('--K', type=int, default=8, help='number of steps in each scale for Glow')
    model_parser.add_argument('--L', type=int, default=3, help='number of scales for Glow')
    model_parser.add_argument('--nolu', action='store_true', help='set to use matrix in 1x1 conv for Glow')

    train_parser = parser.add_argument_group('training', "parameters for training config")
    train_parser.add_argument('--seed', type=int, default=None, help='manuall random seed')
    train_parser.add_argument('--batch_size', type=int, default=128)
    train_parser.add_argument('--gpu', type=str, default='0')
    train_parser.add_argument('--workers', type=int, default=9999,
                              help='how many workers use for dataloader, default is ALL')
    train_parser.add_argument('--steps', type=int, default=30000)
    train_parser.add_argument('--warmup_steps', type=int, default=5000, 
                              help='number of steps for learning rate warmup')
    train_parser.add_argument('--lr', type=float, default=5e-4)
    train_parser.add_argument('--beta1', type=float, default=0.9)
    train_parser.add_argument('--beta2', type=float, default=0.999)

    log_parser = parser.add_argument_group('log', "parameters for log config")
    log_parser.add_argument('--log_step', type=int, default=500)
    log_parser.add_argument('--suffix', type=str, default=None, help='suffix in log folder and model file')

    args = parser.parse_args()

    if not args.custom:
        config = load_config(CONFIG)
        try:
            args = argparse.Namespace(**(config[args.model][args.dataset]))
        except:
            print("Warning: there is no default config for the combination of {} and {}, use custom config instead!".format(
                args.model, args.dataset
            ))
    
    # config gpu
    select_gpus(args.gpu) 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # setup random seed
    seed = args.seed if args.seed else np.random.randint(0, 100000)
    setup_seed(seed) 

    # create output folder
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(MODELDIR):
        os.makedirs(MODELDIR)

    # config dataset
    filenames, model_param, train_loader, val_loader, test_loader = config_dataset(args)

    # config model
    model, model_param = config_model(args, model_param)
    
    model = model.to(device)

    # config optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    schedule = lambda step: min(step / args.warmup_steps, 1.0)
    opt_schedule = torch.optim.lr_scheduler.LambdaLR(optim, schedule)

    writer = SummaryWriter(LOGDIR + '{}'.format(filenames['log_name']))

    # train model
    model, test_loss = train_flow(model, optim, opt_schedule, writer, train_loader, val_loader, test_loader, args)

    dim = model_param['c'] * model_param['h'] * model_param['w']
    torch.save({
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optim.state_dict(),
        "finial_test_loss" : nats2bits(test_loss.item()) / dim,
        "train_time_args" : args,
        "model_parameters" : model_param,
        "seed" : seed,
        "version" : VERSION,
    }, MODELDIR + '{}.pt'.format(filenames['model_name']))