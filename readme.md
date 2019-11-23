## Deep Generative Models
A collection of my PyTorch implementation of several deep generative models.  
This repertory is in progressing, feel free to raise an issue if you find any bug. 

## Requirement  
* PyTorch >= 1.0 (This code was develop on 1.2, but it should also work fine on other version)
* tensorboard (tb-nightly)
* numpy
* matplotlib
* tqdm

## Models
* Auto-regressive
  * [ ] [PixelRNN](http://arxiv.org/abs/1601.06759)
  * [ ] [PixelCNN](http://arxiv.org/abs/1601.06759)
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
  * [ ] [DCGAN](https://arxiv.org/abs/1511.06434)
  * [ ] [InfoGAN](http://arxiv.org/abs/1606.03657)
  * [ ] [WGAN-GP](https://arxiv.org/abs/1704.00028)
  * [ ] [SNGAN](http://arxiv.org/abs/1802.05957)
  * [ ] [SAGAN](http://arxiv.org/abs/1805.08318)
  
## Datasets  
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [SVHN](http://ufldl.stanford.edu/housenumbers/)
* [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Code  
### Structure  
```
/dataset       # default dataset folder
/logs          # default tensorboard log folder
/checkpoints   # default checkpoints folder
/config        # default configurations
/models
    utils.py     # shared utility functions
    modules.py   # shared utility modules
    datasets.py  # data utilities
    /<method>
        train_utils.py    # training procedure
        utils.py          # method's utility functions
        modules.py        # method's utility modules
        <model.py>        # model class
    ......
```
### Train  
Run `python train_<method>.py --dataset <dataset> --model <model>` to train in default configuration.
You can run `python check_default_config.py <method>` to find the default configuration we provide, or just look inside `config` folder.  
If you want to tune some parameters for yourself, pass `--custom` to the training script, run `python train_<method>.py -h` to see all the parameters that you can tune.  
**Note:** All the default configurations are tested on a single RTX 2080Ti GPU with 11G memory, if you cannot run some default configurations (i.e. Glow), please consider reduce the batch size or features in config file or with a custom mode.

### Test  
Run `python test_<model>.py -h` for help.