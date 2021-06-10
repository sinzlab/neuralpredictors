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
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(
            lambda x: not x.startswith("_") and ("gamma" in x or "pool" in x or "positive" in x),
            dir(self),
        ):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


class SpatialXFeatureLinear(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.
    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        normalize=True,
        init_noise=1e-3,
        constrain_pos=False,
        **kwargs,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims
        self.normalize = normalize
        c, w, h = in_shape
        self.spatial = Parameter(torch.Tensor(self.outdims, w, h))
        self.features = Parameter(torch.Tensor(self.outdims, c))
        self.init_noise = init_noise
        self.constrain_pos = constrain_pos
        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)
        self.initialize()

    @property
    def normalized_spatial(self):
        if self.normalize:
            norm = self.spatial.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial) + 1e-6
            weight = self.spatial / norm
        else:
            weight = self.spatial
        if self.constrain_pos:
            positive(weight)
        return weight

    # TODO: Fix weight property -> self.positive is not defined
    @property
    def weight(self):
        if self.positive:
            positive(self.features)
        n = self.outdims
        c, w, h = self.in_shape
        return self.normalized_spatial.view(n, 1, w, h) * self.features.view(n, c, 1, 1)

    def l1(self, average=False):
        n = self.outdims
        c, w, h = self.in_shape
        ret = (
            self.normalized_spatial.view(self.outdims, -1).abs().sum(dim=1, keepdim=True)
            * self.features.view(self.outdims, -1).abs().sum(dim=1)
        ).sum()
        if average:
            ret = ret / (n * c * w * h)
        return ret

    def initialize(self):
        self.spatial.data.normal_(0, self.init_noise)
        self.features.data.normal_(0, self.init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, shift=None):
        if self.constrain_pos:
            positive(self.features)

        y = torch.einsum("ncwh,owh->nco", x, self.normalized_spatial)
        y = torch.einsum("nco,oc->no", y, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        return (
            ("normalized " if self.normalize else "")
            + self.__class__.__name__
            + " ("
            + "{} x {} x {}".format(*self.in_shape)
            + " -> "
            + str(self.outdims)
            + ")"
        )


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
        """Regularization is only applied on the scaled feature weights, not on the bias."""
        if average:
            return (self._source.features * self.alpha).abs().mean()
        else:
            return (self._source.features * self.alpha).abs().sum()

    def initialize(self):
        self.alpha.data.fill_(1.0)
        self.beta.data.fill_(0.0)
