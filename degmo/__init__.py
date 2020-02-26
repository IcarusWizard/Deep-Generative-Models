import os
import os.path as osp

__version__ = 0.1
VERSION = __version__

from degmo import autoregressive, flow, vae, gan
from degmo.datasets import add_dataset, make_dataset

# absolute path for configs
ROOT_PATH = osp.abspath(osp.dirname(__file__))
CONFIG_PATH = osp.join(ROOT_PATH, 'config')

# relative path datasets, checkpoints and logs
# folders will be create under the path you run program
from degmo.datasets.datasets import DATADIR
LOGDIR = 'logs/'
MODELDIR = 'checkpoints/'
