import numpy as np
import torch, torchvision, math
from torch.functional import F
import os
import PIL.Image as Image
from functools import partial

DATADIR = 'dataset/'

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.image_list = [os.path.join(root, filename) for filename in os.listdir(root)]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = Image.open(self.image_list[index])
        if self.transform:
            img = self.transform(img)
        return (img, )

def load_mnist(normalize=False):
    config = {
        "c" : 1,
        "h" : 28,
        "w" : 28,
    }

    transform = [torchvision.transforms.ToTensor()]
    if normalize:
        transform.append(torchvision.transforms.Normalize([0.5], [0.5]))
    transform = torchvision.transforms.Compose(transform)
    train_dataset = torchvision.datasets.MNIST(DATADIR, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(DATADIR, train=False, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (55000, 5000))
    return (train_dataset, val_dataset, test_dataset, config)

def load_bmnist():
    config = {
        "c" : 1,
        "h" : 28,
        "w" : 28,
    }

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: (x > 0).float()),
    ])  
    train_dataset = torchvision.datasets.MNIST(DATADIR, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(DATADIR, train=False, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (55000, 5000))
    return (train_dataset, val_dataset, test_dataset, config)    

def load_svhn(normalize=False):
    config = {
        "c" : 3,
        "h" : 32,
        "w" : 32,
    }

    transform = [torchvision.transforms.ToTensor()]
    if normalize:
        transform.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = torchvision.transforms.Compose(transform)
    train_dataset = torchvision.datasets.SVHN(DATADIR, split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.SVHN(DATADIR, split='test', download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (len(train_dataset) - 5000, 5000))
    return (train_dataset, val_dataset, test_dataset, config)

def load_cifar(normalize=False):
    config = {
        "c" : 3,
        "h" : 32,
        "w" : 32,
    }

    transform = [torchvision.transforms.ToTensor()]
    if normalize:
        transform.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = torchvision.transforms.Compose(transform)
    dataset = torchvision.datasets.CIFAR10(DATADIR, download=True, transform=transform)
    return torch.utils.data.random_split(dataset, (40000, 5000, 5000)) + [config]

def load_celeba(image_size=128, normalize=False):
    config = {
        "c" : 3,
        "h" : image_size,
        "w" : image_size,
    }

    transform = [
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
    ]
    if normalize:
        transform.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = torchvision.transforms.Compose(transform)
    dataset = torchvision.datasets.CelebA(DATADIR, download=True, transform=transform)
    return torch.utils.data.random_split(dataset, (len(dataset) - 2000, 1000, 1000)) + [config]

load_celeba32 = partial(load_celeba, image_size=32)
load_celeba64 = partial(load_celeba, image_size=64)
load_celeba128 = partial(load_celeba, image_size=128)