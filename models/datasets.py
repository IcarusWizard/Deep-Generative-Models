import numpy as np
import torch, torchvision, math
from torch.functional import F
from torch.nn import Parameter, init
import matplotlib.pyplot as plt
import pickle, torch, math, random, os
import PIL.Image as Image

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

def load_mnist():
    transform = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST('dataset', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('dataset', train=False, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (55000, 5000))
    return (train_dataset, val_dataset, test_dataset)

def load_bmnist():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: (x > 0).float()),
    ])  
    train_dataset = torchvision.datasets.MNIST('dataset', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('dataset', train=False, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (55000, 5000))
    return (train_dataset, val_dataset, test_dataset)    

def load_svhn():
    transform = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.SVHN('dataset', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.SVHN('dataset', split='test', download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (len(train_dataset) - 5000, 5000))
    return (train_dataset, val_dataset, test_dataset)

def load_cifar():
    dataset = torchvision.datasets.CIFAR10('dataset', download=True, transform=torchvision.transforms.ToTensor())
    return torch.utils.data.random_split(dataset, (40000, 5000, 5000))

def load_celeba(image_size=128):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CelebA('dataset', download=True, transform=transform)
    return torch.utils.data.random_split(dataset, (len(dataset) - 2000, 1000, 1000))