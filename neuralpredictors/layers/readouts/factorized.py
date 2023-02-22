import numpy as np
import torch
from torch import nn as nn

from .base import Readout


class FullFactorized2d(Readout):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.
    """

    def __init__(
        self,
        in_shape,  # channels, height, width
        outdims,
        bias,
        normalize=True,
        init_noise=1e-3,
        constrain_pos=False,
        positive_weights=False,
        shared_features=None,
        mean_activity=None,
        spatial_and_feature_reg_weight=1.0,
        gamma_readout=None,  # depricated, use feature_reg_weight instead
        **kwargs,
    ):

        super().__init__()

        h, w = in_shape[1:]  # channels, height, width
        self.in_shape = in_shape
        self.outdims = outdims
        self.positive_weights = positive_weights
        self.constrain_pos = constrain_pos
        self.init_noise = init_noise
        self.normalize = normalize
        self.mean_activity = mean_activity
        self.spatial_and_feature_reg_weight = self.resolve_deprecated_gamma_readout(
            spatial_and_feature_reg_weight, gamma_readout
        )

        self._original_features = True
        self.initialize_features(**(shared_features or {}))
        self.spatial = nn.Parameter(torch.Tensor(self.outdims, h, w))

        if bias:
            bias = nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize(mean_activity)

    @property
    def shared_features(self):
        return self._features

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[self.feature_sharing_index, ...]
        else:
            return self._features

    @property
    def weight(self):
        if self.positive_weights:
            self.features.data.clamp_min_(0)
        n = self.outdims
        c, h, w = self.in_shape
        return self.normalized_spatial.view(n, 1, w, h) * self.features.view(n, c, 1, 1)

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

    def regularizer(self, reduction="sum", average=None):
        return self.l1(reduction=reduction, average=average) * self.spatial_and_feature_reg_weight

    def l1(self, reduction="sum", average=None):
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)
        if reduction is None:
            raise ValueError("Reduction of None is not supported in this regularizer")

        n = self.outdims
        c, h, w = self.in_shape
        ret = (
            self.normalized_spatial.view(self.outdims, -1).abs().sum(dim=1, keepdim=True)
            * self.features.view(self.outdims, -1).abs().sum(dim=1)
        ).sum()
        if reduction == "mean":
            ret = ret / (n * c * w * h)
        return ret

    def initialize(self, mean_activity=None):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        if mean_activity is None:
            mean_activity = self.mean_activity
        self.spatial.data.normal_(0, self.init_noise)
        self._features.data.normal_(0, self.init_noise)
        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c = self.in_shape[0]
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (
                    n_match_ids,
                    c,
                ), f"shared features need to have shape ({n_match_ids}, {c})"
                self._features = shared_features
                self._original_features = False
            else:
                self._features = nn.Parameter(
                    torch.Tensor(n_match_ids, c)
                )  # feature weights for each channel of the core
            self.scales = nn.Parameter(torch.Tensor(self.outdims, 1))  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = nn.Parameter(torch.Tensor(self.outdims, c))  # feature weights for each channel of the core
            self._shared_features = False

    def forward(self, x, shift=None, **kwargs):
        if shift is not None:
            raise NotImplementedError("shift is not implemented for this readout")
        if self.constrain_pos:
            self.features.data.clamp_min_(0)

        c, h, w = x.size()[1:]
        c_in, h_in, w_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")

        y = torch.einsum("ncwh,owh->nco", x, self.normalized_spatial)
        y = torch.einsum("nco,oc->no", y, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        c, h, w = self.in_shape
        r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        if self._shared_features:
            r += ", with {} features".format("original" if self._original_features else "shared")
        if self.normalize:
            r += ", normalized"
        else:
            r += ", unnormalized"
        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r


# Classes for backwards compatibility
class SpatialXFeatureLinear(FullFactorized2d):
    pass


class FullSXF(FullFactorized2d):
    pass
