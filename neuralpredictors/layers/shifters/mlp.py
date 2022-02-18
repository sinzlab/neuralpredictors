import logging

import torch
from torch import nn
from torch.nn import ModuleDict
from torch.nn.init import xavier_normal

from .base import Shifter

logger = logging.getLogger(__name__)


class MLP(Shifter):
    def __init__(self, input_features=2, hidden_channels=10, shift_layers=1, **kwargs):
        """
        Multi-layer perceptron shifter
        Args:
            input_features (int): number of input features, defaults to 2.
            hidden_channels (int): number of hidden units.
            shift_layers(int): number of shifter layers (n=1 will correspond to a network without a hidden layer).
            **kwargs:
        """
        super().__init__()

        prev_output = input_features
        feat = []
        for _ in range(shift_layers - 1):
            feat.extend([nn.Linear(prev_output, hidden_channels), nn.Tanh()])
            prev_output = hidden_channels

        feat.extend([nn.Linear(prev_output, 2), nn.Tanh()])
        self.mlp = nn.Sequential(*feat)

    def regularizer(self):
        return 0

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def forward(self, pupil_center, trial_idx=None):
        if trial_idx is not None:
            pupil_center = torch.cat((pupil_center, trial_idx), dim=1)
        if not self.mlp[0].in_features == pupil_center.shape[1]:
            raise ValueError(
                "The expected input shape of the shifter and the shape of the input do not match! "
                "(Maybe due to the appending of trial_idx to pupil_center?)"
            )
        return self.mlp(pupil_center)


class MLPShifter(ModuleDict):
    def __init__(
        self, data_keys, input_channels=2, hidden_channels_shifter=2, shift_layers=1, gamma_shifter=0, **kwargs
    ):
        """
        Args:
            data_keys (list of str): keys of the shifter dictionary, correspond to the data_keys of the nnfabirk dataloaders
            gamma_shifter: weight of the regularizer

            See docstring of base class for the other arguments.
        """
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(k, MLP(input_channels, hidden_channels_shifter, shift_layers))

    def initialize(self, **kwargs):
        logger.info("Ignoring input {} when initializing {}".format(repr(kwargs), self.__class__.__name__))
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_shifter
