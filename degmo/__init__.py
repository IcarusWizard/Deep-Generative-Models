import os
import os.path as osp

from .version import __version__
VERSION = __version__

from degmo import autoregressive, flow, vae, gan
from degmo.data import add_dataset, make_dataset

# absolute path for configs
ROOT_PATH = osp.abspath(osp.dirname(__file__))
CONFIG_PATH = osp.join(ROOT_PATH, 'config')

# relative path datasets, checkpoints and logs
# folders will be create under the path you run program
from degmo.data.datasets import DATADIR
LOGDIR = 'logs/'
MODELDIR = 'checkpoints/'
