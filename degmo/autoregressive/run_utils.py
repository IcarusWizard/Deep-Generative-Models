import torch
import numpy as np

from . import *

def config_model_train(config, model_param):
    if config['model']== 'PixelRNN':
        model_param.update({
            "mode" : config['mode'],
            "features" : config['features'],
            "layers" : config['layers'],
            "post_features" : config['post_features'],
            "bits" : config['bits'],
        })
        model = PixelRNN(**model_param)
    elif config['model']== 'PixelCNN':
        model_param.update({
            "mode" : config['mode'],
            "features" : config['features'],
            "layers" : config['layers'],
            "bits" : config['bits'],
            "kernel_size" : config['kernel_size'],
            "post_features" : config['post_features'],
            "bits" : config['bits'],
        })
        model = PixelCNN(**model_param)
    else:
        raise ValueError('Model {} is not supported!'.format(config['model']))

    return model, model_param