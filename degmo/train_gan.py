import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter

import os, argparse
from tqdm import tqdm

import degmo
from degmo.gan import config_model, train_gan
from degmo.utils import setup_seed, select_gpus, nats2bits, config_dataset, load_config
from degmo import LOGDIR, MODELDIR, VERSION, CONFIG_PATH

CONFIG = os.path.join(CONFIG_PATH, 'gan.json')
LOGDIR = os.path.join(LOGDIR, 'gan')
MODELDIR = os.path.join(MODELDIR, 'gan')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='choose from mnist, svhn, cifar, celeba32, celeba64, celeba128')
    parser.add_argument('--model', type=str, default='GAN',
                        help='choose from GAN, DCGAN, WGAN-GP')
    parser.add_argument('--custom', action='store_true', help='enable custom config')

    model_parser = parser.add_argument_group('model', 'parameters for model config')
    model_parser.add_argument('--latent_dim', type=int, default=32)
    model_parser.add_argument('--discriminator_hidden_layers', type=int, default=2)
    model_parser.add_argument('--discriminator_features', type=int, default=256)
    model_parser.add_argument('--generator_hidden_layers', type=int, default=2)
    model_parser.add_argument('--generator_features', type=int, default=1024)
    model_parser.add_argument('--mode', type=str, default='conv', 
                              help='For WGAN, choose from mlp and conv; For WGAN-GP choose from conv and res')
    model_parser.add_argument('--use_norm_discriminator', type=bool, default=True)
    model_parser.add_argument('--use_norm_generator', type=bool, default=True)
    model_parser.add_argument('--_lambda', type=float, default=10.0)

    train_parser = parser.add_argument_group('training', "parameters for training config")
    train_parser.add_argument('--seed', type=int, default=None, help='manuall random seed')
    train_parser.add_argument('--batch_size', type=int, default=256)
    train_parser.add_argument('--gpu', type=str, default='0')
    train_parser.add_argument('--workers', type=int, default=9999,
                              help='how many workers use for dataloader, default is ALL')
    train_parser.add_argument('--steps', type=int, default=30000)
    train_parser.add_argument('--n_critic', type=int, default=1, 
                              help='number of discriminator update per generator update')
    train_parser.add_argument('--generator_lr', type=float, default=1e-3)
    train_parser.add_argument('--generator_beta1', type=float, default=0.9)
    train_parser.add_argument('--generator_beta2', type=float, default=0.999)
    train_parser.add_argument('--discriminator_lr', type=float, default=1e-3)
    train_parser.add_argument('--discriminator_beta1', type=float, default=0.9)
    train_parser.add_argument('--discriminator_beta2', type=float, default=0.999)
    train_parser.add_argument('--clamp', type=float, default=0.01)

    log_parser = parser.add_argument_group('log', "parameters for log config")
    log_parser.add_argument('--log_step', type=int, default=500, help='log period')
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
    filenames, model_param, train_loader, _, _ = config_dataset(args, normalize=True) # only need train dataset for GAN

    # config model
    model, model_param = config_model(args, model_param)
    model = model.to(device)

    # open log
    writer = SummaryWriter(os.path.join(LOGDIR, '{}'.format(filenames['log_name'])))

    # train model
    model, discriminator_optim, generator_optim = train_gan(model, writer, train_loader, args)

    # save checkpoint
    torch.save({
        "discriminator_state_dict" : model.discriminator.state_dict(),
        "generator_state_dict" : model.generator.state_dict(),
        "train_time_args" : args,
        "model_parameters" : model_param,
        "seed" : seed,
        "version" : VERSION,
    }, os.path.join(MODELDIR,'{}.pt'.format(filenames['model_name'])))