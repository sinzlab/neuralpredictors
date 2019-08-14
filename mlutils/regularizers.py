import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# def laplace():
#     return np.array([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]).astype(np.float32)[None, None, ...]

def laplace():
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)[None, None, ...]


class Laplace(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self,laplace_padding):
        super().__init__()
        self.register_buffer('filter', torch.from_numpy(laplace()))
        if laplace_padding == None:
            self.padding_size = self.filter.shape[-1] // 2
        else:
            self.padding_size = laplace_padding


    def forward(self, x):
        return F.conv2d(x, self.filter, bias=None, padding=self.padding_size)



class LaplaceL2(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self, laplace_padding = None):
        super().__init__()
        self.laplace = Laplace(laplace_padding)


    def forward(self, x):
        ic, oc, k1, k2 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2)).pow(2).mean() / 2
