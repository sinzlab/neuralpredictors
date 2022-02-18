import logging
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from .base import Readout

logger = logging.getLogger(__name__)


class PointPooled2d(Readout):
    def __init__(
        self,
        in_shape,
        outdims,
        pool_steps,
        bias,
        pool_kern,
        init_range,
        align_corners=True,
        mean_activity=None,
        feature_reg_weight=1.0,
        gamma_readout=None,  # depricated, use feature_reg_weight instead
        **kwargs,
    ):
        """
        This readout learns a point in the core feature space for each neuron, with help of torch.grid_sample, that best
        predicts its response. Multiple average pooling steps are applied to reduce search space in each stage and thereby, faster convergence to the best prediction point.

        The readout receives the shape of the core as 'in_shape', number of pooling stages to be performed as 'pool_steps', the kernel size and stride length
        to be used for pooling as 'pool_kern', the number of units/neurons being predicted as 'outdims', 'bias' specifying whether
        or not bias term is to be used and 'init_range' range for initialising the grid with uniform distribution, U(-init_range,init_range).
        The grid parameter contains the normalized locations (x, y coordinates in the core feature space) and is clipped to [-1.1] as it a
        requirement of the torch.grid_sample function. The feature parameter learns the best linear mapping from the pooled feature
        map from a given location to a unit's response with or without an additional elu non-linearity.

        Args:
            in_shape (list): shape of the input feature map [channels, width, height]
            outdims (int): number of output units
            pool_steps (int): number of pooling stages
            bias (bool): adds a bias term
            pool_kern (int): filter size and stride length used for pooling the feature map
            init_range (float): intialises the grid with Uniform([-init_range, init_range])
                                [expected: positive value <=1]
            align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        """
        super().__init__()
        if init_range > 1.0 or init_range <= 0.0:
            raise ValueError("init_range is not within required limit!")
        self._pool_steps = pool_steps
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(feature_reg_weight, gamma_readout)
        self.mean_activity = mean_activity
        self.grid = Parameter(torch.Tensor(1, outdims, 1, 2))  # x-y coordinates for each neuron
        self.features = Parameter(
            torch.Tensor(1, c * (self._pool_steps + 1), 1, outdims)
        )  # weight matrix mapping the core features to the output units

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.pool_kern = pool_kern
        self.avg = nn.AvgPool2d(
            (pool_kern, pool_kern), stride=pool_kern, count_include_pad=False
        )  # setup kernel of size=[pool_kern,pool_kern] with stride=pool_kern
        self.init_range = init_range
        self.align_corners = align_corners
        self.initialize(mean_activity)

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        assert value >= 0 and int(value) - value == 0, "new pool steps must be a non-negative integer"
        if value != self._pool_steps:
            logger.info("Resizing readout features")
            c, w, h = self.in_shape
            self._pool_steps = int(value)
            self.features = Parameter(torch.Tensor(1, c * (self._pool_steps + 1), 1, self.outdims))
            self.features.data.fill_(1 / self.in_shape[0])

    def initialize(self, mean_activity=None):
        """
        Initialize function initialises the grid, features or weights and bias terms.
        """
        if mean_activity is None:
            mean_activity = self.mean_activity
        self.grid.data.uniform_(-self.init_range, self.init_range)
        self.features.data.fill_(1 / self.in_shape[0])
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def feature_l1(self, reduction="sum", average=None):
        """
        Returns l1 regularization term for features.
        Args:
            average(bool): Deprecated (see reduction) if True, use mean of weights for regularization
            reduction(str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        return self.apply_reduction(self.features.abs(), reduction=reduction, average=average)

    def regularizer(self, reduction="sum", average=None):
        return self.feature_l1(reduction=reduction, average=average) * self.feature_reg_weight

    def forward(self, x, shift=None, out_idx=None, **kwargs):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            shift: shifts the location of the grid (from eye-tracking data)
            out_idx: index of neurons to be predicted

        Returns:
            y: neuronal activity
        """
        self.grid.data = torch.clamp(self.grid.data, -1, 1)
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if [c_in, w_in, h_in] != [c, w, h]:
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")

        m = self.pool_steps + 1  # the input feature is considered the first pooling stage
        feat = self.features.view(1, m * c, self.outdims)
        if out_idx is None:
            grid = self.grid
            bias = self.bias
            outdims = self.outdims
        else:
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = self.grid[:, out_idx]
            if self.bias is not None:
                bias = self.bias[out_idx]
            outdims = len(out_idx)

        if shift is None:
            grid = grid.expand(N, outdims, 1, 2)
        else:
            # shift grid based on shifter network's prediction
            grid = grid.expand(N, outdims, 1, 2) + shift[:, None, None, :]

        pools = [F.grid_sample(x, grid, align_corners=self.align_corners)]
        for _ in range(self.pool_steps):
            _, _, w_pool, h_pool = x.size()
            if w_pool * h_pool == 1:
                warnings.warn("redundant pooling steps: pooled feature map size is already 1X1, consider reducing it")
            x = self.avg(x)
            pools.append(F.grid_sample(x, grid, align_corners=self.align_corners))
        y = torch.cat(pools, dim=1)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        r += " and pooling for {} steps\n".format(self.pool_steps)
        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r


class SpatialTransformerPooled3d(Readout):
    def __init__(
        self,
        in_shape,
        outdims,
        pool_steps=1,
        positive=False,
        bias=True,
        init_range=0.05,
        kernel_size=2,
        stride=2,
        grid=None,
        stop_grad=False,
        align_corners=True,
        mean_activity=None,
        feature_reg_weight=1.0,
        gamma_readout=None,  # depricated, use feature_reg_weight instead
    ):
        super().__init__()
        self._pool_steps = pool_steps
        self.in_shape = in_shape
        c, t, w, h = in_shape
        self.outdims = outdims
        self.positive = positive
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(feature_reg_weight, gamma_readout)
        self.mean_activity = mean_activity
        if grid is None:
            self.grid = Parameter(torch.Tensor(1, outdims, 1, 2))
        else:
            self.grid = grid
        self.features = Parameter(torch.Tensor(1, c * (self._pool_steps + 1), 1, outdims))
        self.register_buffer("mask", torch.ones_like(self.features))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.avg = nn.AvgPool2d(kernel_size, stride=stride, count_include_pad=False)
        self.init_range = init_range
        self.initialize(mean_activity)
        self.stop_grad = stop_grad
        self.align_corners = align_corners

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        assert value >= 0 and int(value) - value == 0, "new pool steps must be a non-negative integer"
        if value != self._pool_steps:
            logger.info("Resizing readout features")
            c, t, w, h = self.in_shape
            outdims = self.outdims
            self._pool_steps = int(value)
            self.features = Parameter(torch.Tensor(1, c * (self._pool_steps + 1), 1, outdims))
            self.mask = torch.ones_like(self.features)
            self.features.data.fill_(1 / self.in_shape[0])

    def initialize(self, init_noise=1e-3, grid=True, mean_activity=None):
        if mean_activity is None:
            mean_activity = self.mean_activity
        # randomly pick centers within the spatial map
        self.features.data.fill_(1 / self.in_shape[0])
        if grid:
            self.grid.data.uniform_(-self.init_range, self.init_range)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def feature_l1(self, reduction="sum", average=None, subs_idx=None):
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        return self.apply_reduction(self.features[..., subs_idx].abs(), reduction=reduction, average=average)

    def regularizer(self, reduction="sum", average=None):
        return self.feature_l1(reduction=reduction, average=average) * self.feature_reg_weight

    def reset_fisher_prune_scores(self):
        self._prune_n = 0
        self._prune_scores = self.features.detach() * 0

    def update_fisher_prune_scores(self):
        self._prune_n += 1
        if self.features.grad is None:
            raise ValueError("You need to run backward first")
        self._prune_scores += (0.5 * self.features.grad.pow(2) * self.features.pow(2)).detach()

    @property
    def fisher_prune_scores(self):
        return self._prune_scores / self._prune_n

    def prune(self):
        idx = (self.fisher_prune_scores + 1e6 * (1 - self.mask)).squeeze().argmin(dim=0)
        nt = idx.new
        seq = nt(np.arange(len(idx)))
        self.mask[:, idx, :, seq] = 0
        self.features.data[:, idx, :, seq] = 0

    def forward(self, x, shift=None, subs_idx=None, **kwargs):
        if self.stop_grad:
            x = x.detach()

        self.features.data *= self.mask

        if self.positive:
            self.features.data.clamp_min_(0)
        self.grid.data = torch.clamp(self.grid.data, -1, 1)

        N, c, t, w, h = x.size()
        m = self._pool_steps + 1
        if subs_idx is not None:
            feat = self.features[..., subs_idx].contiguous()
            outdims = feat.size(-1)
            feat = feat.view(1, m * c, outdims)
            grid = self.grid[:, subs_idx, ...]
        else:
            grid = self.grid
            feat = self.features.view(1, m * c, self.outdims)
            outdims = self.outdims

        if shift is None:
            grid = grid.expand(N * t, outdims, 1, 2)
        else:
            grid = grid.expand(N, outdims, 1, 2)
            grid = torch.stack([grid + shift[:, i, :][:, None, None, :] for i in range(t)], 1)
            grid = grid.contiguous().view(-1, outdims, 1, 2)
        z = x.contiguous().transpose(2, 1).contiguous().view(-1, c, w, h)
        pools = [F.grid_sample(z, grid, align_corners=self.align_corners)]
        for i in range(self._pool_steps):
            z = self.avg(z)
            pools.append(F.grid_sample(z, grid, align_corners=self.align_corners))
        y = torch.cat(pools, dim=1)
        y = (y.squeeze(-1) * feat).sum(1).view(N, t, outdims)

        if self.bias is not None:
            if subs_idx is None:
                y = y + self.bias
            else:
                y = y + self.bias[subs_idx]

        return y

    def __repr__(self):
        c, _, w, h = self.in_shape
        r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        if self.stop_grad:
            r += ", stop_grad=True"
        r += "\n"

        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r
