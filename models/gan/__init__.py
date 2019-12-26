from .vanilla_gan import GAN
from .dcgan import DCGAN
from .wgan import WGAN
from .wgangp import WGANGP
from .sngan import SNGAN
from .sagan import SAGAN

from .train_utils import train_gan, config_model

LOGDIR = 'logs/gan/'
MODELDIR = 'checkpoints/gan/'
CONFIG = 'config/gan.json'
VERSION = '0.2'