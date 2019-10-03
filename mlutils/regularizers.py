import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# def laplace():
#     return np.array([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]).astype(np.float32)[None, None, ...]

def laplace():
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)[None, None, ...]


def laplace3d():
    l = np.zeros((3, 3, 3))
    l[1, 1, 1] = -6.
    l[1, 1, 2] = 1.
    l[1, 1, 0] = 1.
    l[1, 0, 1] = 1.
    l[1, 2, 1] = 1.
    l[0, 1, 1] = 1.
    l[2, 1, 1] = 1.
    return l.astype(np.float32)[None, None, ...]

class Laplace(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self, padding=None):
        super().__init__()
        self.register_buffer('filter', torch.from_numpy(laplace()))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding


    def forward(self, x):
        return F.conv2d(x, self.filter, bias=None, padding=self.padding_size)



class LaplaceL2(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace(padding=padding)


    def forward(self, x):
        ic, oc, k1, k2 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2)).pow(2).mean() / 2


class Laplace3d(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('filter', torch.from_numpy(laplace3d()))

    def forward(self, x):
        return F.conv3d(x, self.filter, bias=None)


class LaplaceL23d(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace3d()

    def forward(self, x):
        ic, oc, k1, k2, k3 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2, k3)).pow(2).mean() / 2


class FlatLaplaceL23d(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, x):
        ic, oc, k1, k2, k3 = x.size()
        assert k1 == 1, 'time dimension must be one'
        return self.laplace(x.view(ic * oc, 1, k2, k3)).pow(2).mean() / 2


class LaplaceL1(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self, padding=0):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x):
        ic, oc, k1, k2 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2)).abs().mean()
