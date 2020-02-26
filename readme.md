## Deep Generative Models
A collection of my PyTorch implementation of several deep generative models.  
This repertory is in progressing, feel free to raise an issue if you find any bug. 

## Requirement  
* PyTorch >= 1.0 (This code was develop on 1.3.1, but it should also work fine on other version)
* tensorboard (tb-nightly)
* numpy, scipy (ndarry support)
* matplotlib, moviepy (visualizing result)
* tqdm (progress bar)

## Setup  
Recommend to setup with Anaconda
```
git clone https://github.com/IcarusWizard/Deep-Generative-Models
cd Deep-Generative-Models
pip install -e .
```

## Models
* Auto-regressive
  * [x] [PixelRNN](http://arxiv.org/abs/1601.06759)
  * [x] [PixelCNN](http://arxiv.org/abs/1606.05328)
  * [ ] [PixelSNAIL](http://arxiv.org/abs/1712.09763)
* FLOW
  - [x] [NICE](https://arxiv.org/abs/1410.8516)
  - [x] [RealNVP](http://arxiv.org/abs/1605.08803)
  - [x] [Glow](http://arxiv.org/abs/1807.03039)
* VAE
  * [x] [Vanilla VAE](http://arxiv.org/abs/1312.6114)
  * [x] CNN-VAE
  * [ ] [VQ-VAE](http://arxiv.org/abs/1711.00937)
* GAN
  * [x] [Vanilla GAN](https://arxiv.org/abs/1406.2661)
  * [x] [DCGAN](https://arxiv.org/abs/1511.06434)
  * [ ] [InfoGAN](http://arxiv.org/abs/1606.03657)
  * [x] [WGAN](http://arxiv.org/abs/1701.07875)
  * [x] [WGAN-GP](https://arxiv.org/abs/1704.00028)
  * [x] [SNGAN](http://arxiv.org/abs/1802.05957)
  * [x] [SAGAN](http://arxiv.org/abs/1805.08318)
  
## Datasets  
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [SVHN](http://ufldl.stanford.edu/housenumbers/)
* [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  

If you want to add new dataset, you need to define a creator function which returns three `torch.utils.data.Dataset` for training, validation, testing, and a dict holds the configuration of the dataset (c, h, w). Then you can add your custom loader through:
```
import degmo
def custom_creater():
  ......
  return training_set, validation_set, testing_set, config

degmo.add_dataset('name', custom_creater)
```

## Code  
### Structure  
```
logs/          # default tensorboard log folder
checkpoints/   # default checkpoints folder
degmo/         # main folder
  data/          # dataset functions
  config/        # default configurations
  utils.py       # shared utility functions
  modules.py     # shared utility modules
  datasets.py    # data utilities
  <method>/
      train_utils.py    # training procedure
      test_utils.py     # test functions
      utils.py          # method's utility functions
      modules.py        # method's utility modules
      <model.py>        # model class
    ......
```
### Train  
Run `python -m degmo.train_<method>.py --dataset <dataset> --model <model>` to train in default configuration.
You can run `python -m degmo.check_default_config.py <method>` to find the default configuration we provide, or just look inside `degmo/config` folder.  
If you want to tune some parameters for yourself, pass `--custom` to the training script, run `python -m degmo.train_<method>.py -h` to see all the parameters that you can tune.  
**Note:** All the default configurations are tested on a single RTX 2080Ti GPU with 11G memory, if you cannot run some default configurations (i.e. Glow), please consider reduce the batch size or features in config file or with a custom mode.  

During and after training, you can use `tensorboard --logdir=logs` to monitor progress.

### Test  
Run `python -m degmo.test_<model>.py -h` for help.