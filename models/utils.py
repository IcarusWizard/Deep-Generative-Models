import numpy as np
import torch, torchvision, math
from torch.functional import F
from torch.nn import Parameter, init
import matplotlib.pyplot as plt
import pickle, torch, math, random, os, json
import PIL.Image as Image

from .datasets import *

LOG2PI = 0.5 * np.log(2 * np.pi)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename, eval_mode=False):
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    if eval_mode:
        model.eval()

def step_loader(dataloder):
    data = iter(dataloder)
    while True:
        try:
            x = next(data)
        except:
            data = iter(dataloder)
            x = next(data)
        yield x

def select_gpus(gpus="0"):
    '''
        gpus -> string, examples: "0", "0,1,2"
    ''' 
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

def nats2bits(nats):
    return nats / np.log(2)

def bits2nats(bits):
    return bits * np.log(2)

def logit(p):
    """
        inverse function of sigmoid
        p -> tensor \in (0, 1)
    """
    return torch.log(p) - torch.log(1 - p)

def load_config(filename):
    with open(filename, 'r') as f:
        d = json.load(f)
    return d

def config_dataset(args, batch_size=None):
    workers = min(args.workers, os.cpu_count()) # compute actual workers in use
    
    if batch_size is None:
        batch_size = args.batch_size

    if args.dataset == 'bmnist':
        train_dataset, val_dataset, test_dataset = load_bmnist() 
        filenames = {
            "log_name" : "{}_BMNIST".format(args.model),
            "model_name" : "{}_BMNIST".format(args.model),
        }

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=250)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=250)

        model_param = {
            "c" : 1,
            "h" : 28,
            "w" : 28,
        }

    elif args.dataset == 'mnist':

        train_dataset, val_dataset, test_dataset = load_mnist()
        filenames = {
            "log_name" : "{}_MNIST".format(args.model),
            "model_name" : "{}_MNIST".format(args.model),
        }

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=250)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=250)

        model_param = {
            "c" : 1,
            "h" : 28,
            "w" : 28,
        }

    elif args.dataset == 'svhn':
        train_dataset, val_dataset, test_dataset = load_svhn()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=250)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=250)

        filenames = {
            "log_name" : "{}_SVHN".format(args.model),
            "model_name" : "{}_SVHN".format(args.model),
        }

        model_param = {
            "c" : 3,
            "h" : 32,
            "w" : 32,
        }

    elif args.dataset == 'cifar':
        train_dataset, val_dataset, test_dataset = load_cifar()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=250)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=250)

        filenames = {
            "log_name" : "{}_Cifar10".format(args.model),
            "model_name" : "{}_Cifar10".format(args.model),
        }

        model_param = {
            "c" : 3,
            "h" : 32,
            "w" : 32,
        }

    elif args.dataset == 'celeba32':
        train_dataset, val_dataset, test_dataset = load_celeba(image_size=32)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=250)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=250)

        filenames = {
            "log_name" : "{}_CelebA32".format(args.model),
            "model_name" : "{}_CelebA32".format(args.model),
        }

        model_param = {
            "c" : 3,
            "h" : args.image_size,
            "w" : args.image_size,
        }

    elif args.dataset == 'celeba64':
        train_dataset, val_dataset, test_dataset = load_celeba(image_size=64)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

        filenames = {
            "log_name" : "{}_CelebA{}".format(args.model, args.image_size),
            "model_name" : "{}_CelebA{}".format(args.model, args.image_size),
        }

        model_param = {
            "c" : 3,
            "h" : args.image_size,
            "w" : args.image_size,
        }

    elif args.dataset == 'celeba128':
        train_dataset, val_dataset, test_dataset = load_celeba(image_size=128)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

        filenames = {
            "log_name" : "{}_CelebA128".format(args.model),
            "model_name" : "{}_CelebA128".format(args.model),
        }

        model_param = {
            "c" : 3,
            "h" : args.image_size,
            "w" : args.image_size,
        }

    else:
        raise ValueError('Dataset {} is not supported!'.format(args.dataset))

    if args.suffix:
        for key in filenames.keys():
            filenames[key] += '_{}'.format(args.suffix)

    return filenames, model_param, train_loader, val_loader, test_loader