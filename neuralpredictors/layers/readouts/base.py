import warnings
from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import ModuleDict
from torch.nn import Parameter
from torch.nn import functional as F

from ..constraints import positive


class ConfigurationError(Exception):
    pass


# ------------------ Base Classes -------------------------


class Readout(nn.Module):
    def initialize(self, *args, **kwargs):
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def __repr__(self):
        return super().__repr__() + " [{}]\n".format(self.__class__.__name__)

    def resolve_reduction_method(self, reduction="mean", average=None):
        if average is not None:
            warnings.warn("Use of average is deprecated, Please consider using `reduction` instead", DeprecationWarning)
            reduction = "mean" if average else "sum"
        return reduction

    def apply_reduction(self, x, reduction="mean", average=None):
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)

        if reduction == "mean":
            return x.mean()
        elif reduction == "sum":
            return x.sum()
        elif reduction is None:
            return x
        else:
            raise ValueError(
                f"Reduction method '{reduction}' is not recognized. Valid values are ['mean', 'sum', None]"
            )

    def initialize_bias(self, mean_activity=None):
        if mean_activity is None:
            warnings.warn("Readout is NOT initialized with mean activity but with 0!")
            self.bias.data.fill_(0)
        else:
            self.bias.data = mean_activity


class ClonedReadout(nn.Module):
    """
    This readout clones another readout while applying a linear transformation on the output. Used for MultiDatasets
    with matched neurons where the x-y positions in the grid stay the same but the predicted responses are rescaled due
    to varying experimental conditions.
    """

    def __init__(self, original_readout, **kwargs):
        super().__init__()

        self._source = original_readout
        self.alpha = Parameter(torch.ones(self._source.features.shape[-1]))
        self.beta = Parameter(torch.zeros(self._source.features.shape[-1]))

    def forward(self, x):
        x = self._source(x) * self.alpha + self.beta
        return x

    def feature_l1(self, average=True):
        """ Regularization is only applied on the scaled feature weights, not on the bias."""
        if average:
            return (self._source.features * self.alpha).abs().mean()
        else:
            return (self._source.features * self.alpha).abs().sum()

    def initialize(self, **kwargs):
        self.alpha.data.fill_(1.0)
        self.beta.data.fill_(0.0)
