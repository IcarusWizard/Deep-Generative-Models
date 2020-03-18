import numpy as np
import torch, torchvision, math
from torch.functional import F
from torch.nn import Parameter, init
import matplotlib.pyplot as plt
import pickle, torch, math, random, os, json
import PIL.Image as Image

import degmo
from degmo.data import make_dataset

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

def config_dataset(config, batch_size=None, **kargs):
    workers = min(config['workers'], os.cpu_count()) # compute actual workers in use
    
    if batch_size is None:
        batch_size = config['batch_size']

    train_dataset, val_dataset, test_dataset, model_param = make_dataset(config['dataset'], **kargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)

    filenames = {
        "log_name" : "{}_{}".format(config['model'], config['dataset']),
        "model_name" : "{}_{}".format(config['model'], config['dataset']),
    }

    if config['suffix']:
        for key in filenames.keys():
            filenames[key] += '_{}'.format(config['suffix'])

    return filenames, model_param, train_loader, val_loader, test_loader