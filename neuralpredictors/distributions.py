import numpy as np
import torch
from torch.distributions import Distribution, Normal


class ExponentialMixture(Distribution):
    def __init__(self, fraction_on, considered_zero=1e-2):
        super().__init__()
        self.fraction_on = fraction_on
        self.rate = -np.log(considered_zero) / considered_zero

    @staticmethod
    def p_exp(x, rate):
        return rate * torch.exp(-rate * x)

    def log_prob(self, value):
        q = self.fraction_on * self.p_exp(value, 1) + (1 - self.fraction_on) * self.p_exp(value, self.rate)
        return torch.log(q).sum(dim=-1)


class GaussianMixture(Distribution):
    def __init__(self, fraction_on):
        super().__init__()
        self.fraction_on = fraction_on
        self.mix_off = Normal(-1, 0.1)
        self.mix_on = Normal(1, 2)

    def log_prob(self, value):
        lon = self.mix_on.log_prob(value) + torch.log(self.fraction_on)
        loff = self.mix_off.log_prob(value) + torch.log(1 - self.fraction_on)
        return torch.logsumexp(torch.stack((lon, loff), dim=0), dim=0).sum(dim=-1)
