from .nice import NICE
from .realnvp import RealNVP2D
from .glow import GLOW

from .train_utils import config_model, train_flow

LOGDIR = 'logs/flow/'
MODELDIR = 'checkpoints/flow/'
VERSION = '0.1'