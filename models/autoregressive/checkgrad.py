from models.autoregressive.pixelrnn import PixelRNN
from models.autoregressive.pixelcnn import PixelCNN

import torch
import matplotlib.pyplot as plt
import numpy as np

def show(img):
    img = img.permute(0, 2, 3, 1).numpy()[0]
    mg = np.abs(img)
    img = img / np.max(img)
    plt.imshow(img)
    plt.show()

# model = PixelRNN(3, 32, 32, mode='bi', layers=5)
model = PixelCNN(3, 32, 32, "res", 64, layers=16)

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

