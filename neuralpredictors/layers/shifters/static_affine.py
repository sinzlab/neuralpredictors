from .base import Shifter

import torch
from torch import nn
from torch.nn import ModuleDict

import logging
logger = logging.getLogger(__name__)


class StaticAffine2dShifter(Shifter, ModuleDict):
    def __init__(self, data_keys, input_channels, bias=True, gamma_shifter=0, **kwargs):
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(k, StaticAffine2d(input_channels, 2, bias=bias))

    def initialize(self, bias=None):
        for k in self:
            if bias is not None:
                self[k].initialize(bias=bias[k])
            else:
                self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].weight.pow(2).mean() * self.gamma_shifter


class StaticAffine2d(nn.Linear):
    def __init__(self, input_channels, output_channels, bias=True, **kwargs):
        super().__init__(input_channels, output_channels, bias=bias)

    def forward(self, x):
        x = super().forward(x)
        return torch.tanh(x)

    def initialize(self, bias=None):
        self.weight.data.normal_(0, 1e-6)
        if self.bias is not None:
            if bias is not None:
                logger.info('Setting bias to predefined value')
                self.bias.data = bias
            else:
                self.bias.data.normal_(0, 1e-6)