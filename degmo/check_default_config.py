import sys
import os

from degmo import CONFIG_PATH
from degmo.utils import load_config

method = sys.argv[1]

CONFIG = os.path.join(CONFIG_PATH, '{}.json'.format(method))
config = load_config(CONFIG)

for model in config.keys():
    print("{} :".format(model))
    for dataset in config[model].keys():
        print("\t{}".format(dataset))