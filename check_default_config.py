import sys
from models.utils import load_config

method = sys.argv[1]

if method == 'autoregressive':
    from models.autoregressive import CONFIG
elif method == 'flow':
    from models.flow import CONFIG
elif method == 'vae':
    from models.vae import CONFIG
elif method == 'gan':
    from models.gan import CONFIG
else:
    raise ValueError("{} is not a name of supported method!".format(method))

config = load_config(CONFIG)

for model in config.keys():
    print("{} :".format(model))
    for dataset in config[model].keys():
        print("\t{}".format(dataset))