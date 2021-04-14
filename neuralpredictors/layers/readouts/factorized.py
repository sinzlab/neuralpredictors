from warnings import warn
import torch
from torch import nn as nn
from torch.nn import Parameter
from .base import Readout


class SpatialXFeatureLinear(Readout):
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
            positive_weights=False,
            mean_activity=None,
            reg_weight=1.,
            **kwargs,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims
        self.normalize = normalize
        self.positive_weights = positive_weights
        self.reg_weight=reg_weight
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
        self.initialize(mean_activity)

    @property
    def normalized_spatial(self):
        """
        Normalize the spatial mask
        """
        if self.normalize:
            norm = self.spatial.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial) + 1e-6
            weight = self.spatial / norm
        else:
            weight = self.spatial
        if self.constrain_pos:
            weight.data.clamp_min_(0)
        return weight

    @property
    def weight(self):
        if self.positive_weights:
            self.features.data.clamp_min_(0)
        n = self.outdims
        c, w, h = self.in_shape
        return self.normalized_spatial.view(n, 1, w, h) * self.features.view(n, c, 1, 1)

    def regularizer(self, reduction="sum", average=None):
        #TODO: the default for reduction here is different than for the other readouts. -> break backwards compatibility or break consistency between readouts?
        return self.l1(reduction=reduction, average=average) * self.reg_weight

    def l1(self, reduction="sum", average=None):
        reduction = self.reduction_method(reduction=reduction, average=average)
        if reduction is None:
            raise ValueError("Reduction of None is not supported in this regularizer")

        n = self.outdims
        c, w, h = self.in_shape
        ret = (
                self.normalized_spatial.view(self.outdims, -1).abs().sum(dim=1, keepdim=True)
                * self.features.view(self.outdims, -1).abs().sum(dim=1)
        ).sum()
        if reduction == 'mean':
            ret = ret / (n * c * w * h)
        return ret

    def initialize(self, mean_activity=None):
        self.spatial.data.normal_(0, self.init_noise)
        self.features.data.normal_(0, self.init_noise)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def forward(self, x, shift=None):
        if self.constrain_pos:
            self.features.data.clamp_min_(0)

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