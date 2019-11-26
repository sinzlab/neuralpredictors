from torch import nn as nn
from torch.nn import functional as F
import torch



def elu1(x):
    return F.elu(x, inplace=True) + 1.


class Elu1(nn.Module):
    """
    Elu activation function shifted by 1 to ensure that the
    output stays positive. That is:
    Elu1(x) = Elu(x) + 1
    """

    def forward(self, x):
        return elu1(x)


def log1exp(x):
    return torch.log(1. + torch.exp(x))


class Log1Exp(nn.Module):
    def forward(self, x):
        return log1exp(x)

