from .registry import add_dataset, make_dataset

add_dataset('bmnist', 'degmo.data.datasets:load_bmnist')
add_dataset('mnist', 'degmo.data.datasets:load_mnist')
add_dataset('cifar', 'degmo.data.datasets:load_cifar')
add_dataset('svhn', 'degmo.data.datasets:load_svhn')
add_dataset('celeba', 'degmo.data.datasets:load_celeba')
add_dataset('celeba32', 'degmo.data.datasets:load_celeba32')
add_dataset('celeba64', 'degmo.data.datasets:load_celeba64')
add_dataset('celeba128', 'degmo.data.datasets:load_celeba128')