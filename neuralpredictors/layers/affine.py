import torch
from torch import nn


class Bias2DLayer(nn.Module):
    """
    Bias per channel for a given 2d input.
    """

    def __init__(self, channels, initial=0, **kwargs):
        """
        Args:
            channels (int): number of channels in the input.
            initial (int, optional): intial value. Defaults to 0.
        """
        super().__init__(**kwargs)

        self.bias = torch.nn.Parameter(torch.empty((1, channels, 1, 1)).fill_(initial))

    def forward(self, x):
        return x + self.bias


class Scale2DLayer(nn.Module):
    """
    Scale per channel for a given 2d input.
    """

    def __init__(self, channels, initial=1, **kwargs):
        """
        Args:
            channels (int): number of channels in the input.
            initial (int, optional): intial value. Defaults to 1.
        """
        super().__init__(**kwargs)

        self.scale = torch.nn.Parameter(torch.empty((1, channels, 1, 1)).fill_(initial))

    def forward(self, x):
        return x * self.scale
