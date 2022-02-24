import logging
import warnings

import torch
from torch import nn
from torch.nn import ModuleDict

from .base import Shifter

logger = logging.getLogger(__name__)


class StaticAffine2d(nn.Linear):
    def __init__(self, input_channels=2, output_channels=2, bias=True):
        """
        A simple FC network with bias between input and output channels without a hidden layer.
        Args:
            input_channels (int): number of input channels.
            output_channels (int): number of output channels.
            bias (bool): Adds a bias parameter if True.
        """
        super().__init__(input_channels, output_channels, bias=bias)

    def forward(self, x, trial_idx=None):
        if trial_idx is not None:
            warnings.warn(
                "Trial index was passed but is not used because this shifter network does not support trial indexing."
            )
        x = super().forward(x)
        return torch.tanh(x)

    def initialize(self, bias=None):
        self.weight.data.normal_(0, 1e-6)
        if self.bias is not None:
            if bias is not None:
                logger.info("Setting bias to predefined value")
                self.bias.data = bias
            else:
                self.bias.data.normal_(0, 1e-6)

    def regularizer(self):
        return self.weight.pow(2).mean()


class StaticAffine2dShifter(ModuleDict):
    def __init__(self, data_keys, input_channels=2, output_channels=2, bias=True, gamma_shifter=0):
        """
        Args:
            data_keys (list of str): keys of the shifter dictionary, correspond to the data_keys of the nnfabirk dataloaders
            gamma_shifter: weight of the regularizer

            See docstring of base class for the other arguments.
        """
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(k, StaticAffine2d(input_channels, output_channels, bias=bias))

    def initialize(self, bias=None):
        for k in self:
            if bias is not None:
                self[k].initialize(bias=bias[k])
            else:
                self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_shifter
