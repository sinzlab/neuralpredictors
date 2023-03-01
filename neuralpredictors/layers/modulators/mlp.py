import warnings
from collections import OrderedDict

import torch
from torch import nn
from torch.nn.init import xavier_normal


class MLP(nn.Module):
    def __init__(
        self, neurons, input_channels=3, hidden_channels=10, layers=2, bias=True, n_parameters_to_modulate=1, **kwargs
    ):
        super().__init__()
        warnings.warn("Ignoring input {} when creating {}".format(repr(kwargs), self.__class__.__name__))
        self.n_parameters_to_modulate = n_parameters_to_modulate
        self.modulator_networks = nn.ModuleList()
        for _ in range(self.n_parameters_to_modulate):

            prev_output = input_channels
            feat = []
            for _ in range(layers - 1):
                feat.extend([nn.Linear(prev_output, hidden_channels, bias=bias), nn.ReLU()])
                prev_output = hidden_channels

            feat.extend([nn.Linear(prev_output, neurons, bias=bias), nn.ReLU()])
            self.modulator_networks.append(nn.Sequential(*feat))

    def regularizer(self):
        return self.linear.weight.abs().mean()

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def forward(self, x, behavior):
        mods = []
        for network in self.modulator_networks:
            # Make modulation positive. Exponential would result in exploding modulation -> Elu+1
            mods.append(nn.functional.elu(network(behavior)) + 1)
        mods = mods[0] if self.n_parameters_to_modulate == 1 else torch.stack(mods)
        return x * mods


class StaticModulator(torch.nn.ModuleDict):
    _base_modulator = None

    def __init__(
        self,
        n_neurons,
        input_channels=3,
        hidden_channels=5,
        layers=2,
        gamma_modulator=0,
        bias=True,
        n_parameters_to_modulate=1,
        **kwargs
    ):
        warnings.warn("Ignoring input {} when creating {}".format(repr(kwargs), self.__class__.__name__))
        super().__init__()
        self.gamma_modulator = gamma_modulator
        for k, n in n_neurons.items():
            if isinstance(input_channels, OrderedDict):
                ic = input_channels[k]
            else:
                ic = input_channels
            self.add_module(
                k,
                self._base_modulator(
                    n, ic, hidden_channels, layers=layers, bias=bias, n_parameters_to_modulate=n_parameters_to_modulate
                ),
            )

    def initialize(self):
        for k, mu in self.items():
            self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_modulator


class MLPModulator(StaticModulator):
    _base_modulator = MLP


def NoModulator(*args, **kwargs):
    """
    Dummy function to create an object that returns None
    Args:
        *args:   will be ignored
        *kwargs: will be ignored
    Returns:
        None
    """
    return None
