import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter

import os, argparse
from tqdm import tqdm

import degmo
from degmo.vae import config_model, train_vae
from degmo.utils import setup_seed, select_gpus, nats2bits, config_dataset, load_config
from degmo import LOGDIR, MODELDIR, VERSION, CONFIG_PATH

CONFIG = os.path.join(CONFIG_PATH, 'vae.json')
LOGDIR = os.path.join(LOGDIR, 'vae')
MODELDIR = os.path.join(MODELDIR, 'vae')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='choose from bmnist, mnist, svhn, cifar, celeba32, celeba64, celeba128')
    parser.add_argument('--model', type=str, default='VAE',
                        help='choose from VAE, FVAE, VQ-VAE')
    parser.add_argument('--custom', action='store_true', help='enable custom config')

    model_parser = parser.add_argument_group('model', 'parameters for model config')
    model_parser.add_argument('--latent_dim', type=int, default=2)
    model_parser.add_argument('--output_type', type=str, default='gauss',
                              help='the output mode for vae decoder, choose from fix_std, gauss, bernoulli')
    model_parser.add_argument('--use_mce', action='store_true', help='use Mento Carlo Estimation to compute kl divergence')
    model_parser.add_argument('--network_type', type=str, default='conv')
    model_parser.add_argument('--hidden_layers', type=int, default=5)
    model_parser.add_argument('--features', type=int, default=512)
    model_parser.add_argument('--conv_features', type=int, default=256)
    model_parser.add_argument('--down_sampling', type=int, default=2)
    model_parser.add_argument('--res_layers', nargs='+', type=int, default=[0])
    model_parser.add_argument('--use_batchnorm', action='store_true')
    model_parser.add_argument('--k', type=int, default=512)
    model_parser.add_argument('--d', type=int, default=64)
    model_parser.add_argument('--beta', type=float, default=0.25)
    model_parser.add_argument('--flow_features', type=int, default=64)
    model_parser.add_argument('--flow_hidden_layers', type=int, default=3)
    model_parser.add_argument('--flow_num_transformation', type=int, default=3)

    train_parser = parser.add_argument_group('training', "parameters for training config")
    train_parser.add_argument('--seed', type=int, default=None, help='manuall random seed')
    train_parser.add_argument('--batch_size', type=int, default=256)
    train_parser.add_argument('--gpu', type=str, default='0')
    train_parser.add_argument('--workers', type=int, default=9999,
                              help='how many workers use for dataloader, default is ALL')
    train_parser.add_argument('--steps', type=int, default=30000)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--beta1', type=float, default=0.9)
    train_parser.add_argument('--beta2', type=float, default=0.999)

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

    assert not (args.dataset == 'bmnist') ^ (args.output_type == 'bernoulli'), 'bmnist can only use with bernoulli output type'
    
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

    # open log
    writer = SummaryWriter(os.path.join(LOGDIR, '{}'.format(filenames['log_name'])))

    # train model
    model, test_loss = train_vae(model, optim, writer, train_loader, val_loader, test_loader, args)

    # save checkpoint
    torch.save({
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optim.state_dict(),
        "finial_test_loss" : nats2bits(test_loss.item()),
        "train_time_args" : args,
        "model_parameters" : model_param,
        "seed" : seed,
        "version" : VERSION,
    }, os.path.join(MODELDIR, '{}.pt'.format(filenames['model_name'])))