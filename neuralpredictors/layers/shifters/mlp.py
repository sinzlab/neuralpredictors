from .base import Shifter

from torch import nn
from torch.nn.init import xavier_normal
from torch.nn import ModuleDict

import logging
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, input_features=2, hidden_channels=10, shift_layers=1, **kwargs):
        super().__init__()

        feat = []
        if shift_layers > 1:
            feat = [nn.Linear(input_features, hidden_channels), nn.Tanh()]
        else:
            hidden_channels = input_features

        for _ in range(shift_layers - 2):
            feat.extend([nn.Linear(hidden_channels, hidden_channels), nn.Tanh()])

        feat.extend([nn.Linear(hidden_channels, 2), nn.Tanh()])
        self.mlp = nn.Sequential(*feat)

    def regularizer(self):
        return 0

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def forward(self, input):
        return self.mlp(input)


class MLPShifter(Shifter, ModuleDict):
    def __init__(self, data_keys, input_channels=2, hidden_channels_shifter=2,
                 shift_layers=1, gamma_shifter=0, **kwargs):
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(k, MLP(input_channels, hidden_channels_shifter, shift_layers))

    def initialize(self, **kwargs):
        logger.info('Ignoring input {} when initializing {}'.format(repr(kwargs), self.__class__.__name__))
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_shifter