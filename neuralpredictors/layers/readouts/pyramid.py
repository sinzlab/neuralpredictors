import warnings

import numpy as np
import torch
from torch import nn as nn
from torch.nn import Parameter
from torch.nn import functional as F

from .base import Readout


class Pyramid(nn.Module):
    _filter_dict = {
        "gauss5x5": np.float32(
            [
                [0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                [0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
            ]
        ),
        "gauss3x3": np.float32([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]]),
        "laplace5x5": np.outer(np.float32([1, 4, 6, 4, 1]), np.float32([1, 4, 6, 4, 1])) / 256,
    }

    def __init__(self, scale_n=4, type="gauss5x5", downsample=True):
        """
        Setup Laplace image pyramid
        Args:
            scale_n: number of Laplace pyramid layers to construct
            type: type of Gaussian filter used in pyramid construction. Valid options are: 'gauss5x5', 'gauss3x3', and 'laplace5x5'
            downsample: whether to downsample the image in each layer. Defaults to True
        """
        super().__init__()
        self.type = type
        self.downsample = downsample
        h = self._filter_dict[type]
        self.register_buffer("filter", torch.from_numpy(h))
        self.scale_n = scale_n
        self._kern = h.shape[0]
        self._pad = self._kern // 2
        self._filter_cache = None

    def lap_split(self, img):
        N, c, h, w = img.size()
        if self._filter_cache is not None and self._filter_cache.size(0) == c:
            filter = self._filter_cache
        else:
            filter = self.filter.expand(c, 1, self._kern, self._kern).contiguous()
            self._filter_cache = filter

        # the necessary output padding depends on even/odd of the dimension
        output_padding = (h + 1) % 2, (w + 1) % 2

        smooth = F.conv2d(img, filter, padding=self._pad, groups=c)
        if self.downsample:
            lo = smooth[:, :, ::2, ::2]
            lo2 = 4 * F.conv_transpose2d(
                lo,
                filter,
                stride=2,
                padding=self._pad,
                output_padding=output_padding,
                groups=c,
            )
        else:
            lo = lo2 = smooth

        hi = img - lo2

        return lo, hi

    def forward(self, img, **kwargs):
        levels = []
        for i in range(self.scale_n):
            img, hi = self.lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels

    def __repr__(self):
        return "Pyramid(scale_n={scale_n}, padding={_pad}, downsample={downsample}, type={type})".format(
            **self.__dict__
        )


class PointPyramid2d(Readout):
    def __init__(
        self,
        in_shape,
        outdims,
        scale_n,
        positive,
        bias,
        init_range,
        downsample,
        type,
        align_corners=True,
        mean_activity=None,
        feature_reg_weight=1.0,
        gamma_readout=None,  # depricated, use feature_reg_weight instead
        **kwargs,
    ):
        super().__init__()
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.positive = positive
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(feature_reg_weight, gamma_readout)
        self.mean_activity = mean_activity
        self.gauss_pyramid = Pyramid(scale_n=scale_n, downsample=downsample, type=type)
        self.grid = Parameter(torch.Tensor(1, outdims, 1, 2))
        self.features = Parameter(torch.Tensor(1, c * (scale_n + 1), 1, outdims))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)
        self.init_range = init_range
        self.align_corners = align_corners
        self.initialize(mean_activity)

    def initialize(self, mean_activity=None):
        if mean_activity is None:
            mean_activity = self.mean_activity
        self.grid.data.uniform_(-self.init_range, self.init_range)
        self.features.data.fill_(1 / self.in_shape[0])
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def group_sparsity(self, group_size):
        f = self.features.size(1)
        n = f // group_size
        ret = 0
        for chunk in range(0, f, group_size):
            ret = ret + (self.features[:, chunk : chunk + group_size, ...].pow(2).mean(1) + 1e-12).sqrt().mean() / n
        return ret

    def feature_l1(self, reduction="sum", average=None):
        return self.apply_reduction(self.features.abs(), reduction=reduction, average=average)

    def regularizer(self, reduction="sum", average=None):
        return self.feature_l1(reduction=reduction, average=average) * self.feature_reg_weight

    def forward(self, x, shift=None, **kwargs):
        if self.positive:
            self.features.data.clamp_min_(0)
        self.grid.data = torch.clamp(self.grid.data, -1, 1)
        N, c, w, h = x.size()
        m = self.gauss_pyramid.scale_n + 1
        feat = self.features.view(1, m * c, self.outdims)

        if shift is None:
            grid = self.grid.expand(N, self.outdims, 1, 2)
        else:
            grid = self.grid.expand(N, self.outdims, 1, 2) + shift[:, None, None, :]

        pools = [F.grid_sample(xx, grid, align_corners=self.align_corners) for xx in self.gauss_pyramid(x)]
        y = torch.cat(pools, dim=1).squeeze(-1)
        y = (y * feat).sum(1).view(N, self.outdims)

        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"

        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r
