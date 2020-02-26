from degmo.datasets.registry import add_dataset, make_dataset
from .datasets import *

add_dataset('bmnist', load_bmnist)
add_dataset('mnist', load_mnist)
add_dataset('cifar', load_cifar)
add_dataset('svhn', load_svhn)
add_dataset('celeba32', load_celeba32)
add_dataset('celeba64', load_celeba64)
add_dataset('celeba128', load_celeba128)