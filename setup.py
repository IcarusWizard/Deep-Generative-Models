from os.path import join, dirname, realpath
from setuptools import setup
import sys

from degmo import __version__

assert sys.version_info.major == 3, \
    "This repo is designed to work with Python 3." \
    + "Please install it before proceeding."

setup(
    name='degmo',
    py_modules=['degmo'],
    version=__version__,
    install_requires=[
        'torch>=1.0',
        'torchvision',
        'numpy',
        'scipy',
        'moviepy',
        'matplotlib',
        'tb_nightly',
        'tqdm'
    ],
    description="Collection of PyTorch implementation for several deep generative models.",
    author="Xingyuan Zhang",
)