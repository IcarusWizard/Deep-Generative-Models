import torch
from .modules import AffineCoupling1D

class FlowDistribution1D(torch.nn.Module):
    def __init__(self, dim, num_transfrom, features=128, hidden_layers=3):
        """
            This a general distribution based on Flow
        """
        super().__init__()
        self.dim = dim

        self.couplings = torch.nn.ModuleList([AffineCoupling1D(dim, features, hidden_layers) for _ in range(num_transfrom)])

        self.prior = torch.distributions.Normal(0, 1)

    def forward(self, x):
        # forward is a inference path
        z, logdet = x, 0

        for coupling in self.couplings:
            z, _logdet = coupling(z)
            logdet += _logdet

        return z, logdet + torch.sum(self.prior.log_prob(z), dim=1)

    def log_prob(self, x):
        _, log_prob = self.forward(x)
        return log_prob

    def sample(self, num):
        device = next(self.parameters()).device

        z = self.prior.sample((num, self.dim)).to(device)
        
        x = z

        for coupling in reversed(self.couplings):
            x = coupling.backward(x)

        return x

    