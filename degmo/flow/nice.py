import torch

from ..utils import bits2nats
from .modules import Reshape, NICEAdditiveCoupling, Rescale, CoupleSigmoid
from .trainer import FlowTrainer

class NICE(torch.nn.Module):
    def __init__(self, c=3, h=32, w=32, features=1000, hidden_layers=5, bits=8):
        super().__init__()
        self.z_size = c * h * w
        self.bits = bits

        self.reshape = Reshape(c, h, w)
        self.couplings = torch.nn.ModuleList([NICEAdditiveCoupling(self.z_size, features, hidden_layers) for _ in range(4)])
        self.rescale = Rescale(self.z_size)
        self.logistic = CoupleSigmoid()
    
    def forward(self, x, reverse=False, average=True):
        if reverse:
            z = x
            z = self.logistic.backward(z)
            z = self.rescale.backward(z)
            for coupling in reversed(self.couplings):
                z = coupling.backward(z)
            return self.reshape.backward(z)
        else:
            z = self.reshape(x)
            LL = 0
            
            for coupling in self.couplings:
                z, ll = coupling(z)
                LL = LL + ll

            z, ll = self.rescale(z)
            LL = LL + ll

            z, ll = self.logistic(z)
            LL = LL + ll

            LL = LL + 0 # uniform prior

            NLL = - LL + bits2nats(self.bits) * self.z_size

            if average:
                NLL = torch.mean(NLL)

            return z, NLL

    def x2z(self, x):
        z, _ = self.forward(x)

        return z

    def z2x(self, z):
        return self.forward(z, reverse=True)

    def sample(self, num_samples=1000, temperature=1.0):
        device = next(self.parameters()).device
        
        z = torch.rand(num_samples, self.z_size, device=device, dtype=torch.float32) * temperature

        return self.z2x(z)

    def get_trainer(self):
        return FlowTrainer