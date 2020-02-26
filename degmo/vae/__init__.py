from .train_utils import train_vae, config_model

from .vae import VAE
from .convvae import CONV_VAE

LOGDIR = 'logs/vae/'
MODELDIR = 'checkpoints/vae/'
CONFIG = 'config/vae.json'
VERSION = '0.1'