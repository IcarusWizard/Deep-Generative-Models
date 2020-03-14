from degmo.autoregressive.pixelrnn import PixelRNN
from degmo.autoregressive.pixelcnn import PixelCNN

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

def show(img):
    img = img.permute(0, 2, 3, 1).numpy()[0]
    mg = np.abs(img)
    img = np.max(img, axis=2)
    img = img / np.max(img)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    if args.model == 'bi-rnn':
        model = PixelRNN(3, 32, 32, mode='bi', layers=3)
    elif args.model == 'row-rnn':
        model = PixelRNN(3, 32, 32, mode='row', layers=3)
    elif args.model == 'res-cnn':
        model = PixelCNN(3, 32, 32, "res", 64, layers=5)
    elif args.model == 'gate-cnn':
        model = PixelCNN(3, 32, 32, "gate", 64, layers=5)

    x = torch.rand(1, 3, 32, 32, requires_grad=True)

    r = model.get_distribution(x)[0]

    g = model.get_distribution(x)[1]

    b = model.get_distribution(x)[2]

    out_grad = torch.zeros_like(r)
    out_grad[:, :, 16, 16] = 1

    grad_g = torch.autograd.grad(g, x, out_grad)[0]

    grad_r = torch.autograd.grad(r, x, out_grad)[0]

    grad_b = torch.autograd.grad(b, x, out_grad)[0]

    show(grad_r)

    show(grad_g)

    show(grad_b)

    print('r : {}'.format(grad_r[0, :, 16, 16]))
    print('g : {}'.format(grad_g[0, :, 16, 16]))
    print('b : {}'.format(grad_b[0, :, 16, 16]))

