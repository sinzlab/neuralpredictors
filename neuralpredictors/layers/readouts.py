import warnings
from collections import OrderedDict

import numpy as np
from scipy.stats import ortho_group
import torch
from torch import nn as nn
from torch.nn import ModuleDict
from torch.nn import Parameter
from torch.nn import functional as F

from ..constraints import positive
from ..utils import BiasNet


class ConfigurationError(Exception):
    pass


# ------------------ Base Classes -------------------------


class Readout:
    def initialize(self, *args, **kwargs):
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(
            lambda x: not x.startswith("_") and ("gamma" in x or "pool" in x or "positive" in x), dir(self)
        ):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


class SpatialXFeatureLinear(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.
    """

    def __init__(self, in_shape, outdims, bias, normalize=True, init_noise=1e-3, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims
        self.normalize = normalize
        c, w, h = in_shape
        self.spatial = Parameter(torch.Tensor(self.outdims, w, h))
        self.features = Parameter(torch.Tensor(self.outdims, c))
        self.init_noise = init_noise
        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)
        self.initialize()

    @property
    def normalized_spatial(self):
        positive(self.spatial)
        if self.normalize:
            norm = self.spatial.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial) + 1e-6
            weight = self.spatial / norm
        else:
            weight = self.spatial
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


class ClonedReadout(Readout, nn.Module):
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

    def initialize(self):
        self.alpha.data.fill_(1.0)
        self.beta.data.fill_(0.0)


class PointPooled2d(nn.Module):
    def __init__(self, in_shape, outdims, pool_steps, bias, pool_kern, init_range, align_corners=True, **kwargs):
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
        self.initialize()

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        assert value >= 0 and int(value) - value == 0, "new pool steps must be a non-negative integer"
        if value != self._pool_steps:
            print("Resizing readout features")
            c, w, h = self.in_shape
            self._pool_steps = int(value)
            self.features = Parameter(torch.Tensor(1, c * (self._pool_steps + 1), 1, self.outdims))
            self.features.data.fill_(1 / self.in_shape[0])

    def initialize(self):
        """
        Initialize function initialises the grid, features or weights and bias terms.
        """
        self.grid.data.uniform_(-self.init_range, self.init_range)
        self.features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def feature_l1(self, average=True):
        """
        Returns l1 regularization term for features.
        Args:
            average(bool): if True, use mean of weights for regularization

        """
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def forward(self, x, shift=None, out_idx=None):
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


class SpatialTransformerPooled3d(nn.Module):
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
    ):
        super().__init__()
        self._pool_steps = pool_steps
        self.in_shape = in_shape
        c, t, w, h = in_shape
        self.outdims = outdims
        self.positive = positive
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
        self.initialize()
        self.stop_grad = stop_grad
        self.align_corners = align_corners

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        assert value >= 0 and int(value) - value == 0, "new pool steps must be a non-negative integer"
        if value != self._pool_steps:
            print("Resizing readout features")
            c, t, w, h = self.in_shape
            outdims = self.outdims
            self._pool_steps = int(value)
            self.features = Parameter(torch.Tensor(1, c * (self._pool_steps + 1), 1, outdims))
            self.mask = torch.ones_like(self.features)
            self.features.data.fill_(1 / self.in_shape[0])

    def initialize(self, init_noise=1e-3, grid=True):
        # randomly pick centers within the spatial map

        self.features.data.fill_(1 / self.in_shape[0])
        if self.bias is not None:
            self.bias.data.fill_(0)
        if grid:
            self.grid.data.uniform_(-self.init_range, self.init_range)

    def feature_l1(self, average=True, subs_idx=None):
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        if average:
            return self.features[..., subs_idx].abs().mean()
        else:
            return self.features[..., subs_idx].abs().sum()

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

    def forward(self, x, shift=None, subs_idx=None):
        if self.stop_grad:
            x = x.detach()

        self.features.data *= self.mask

        if self.positive:
            positive(self.features)
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
                lo, filter, stride=2, padding=self._pad, output_padding=output_padding, groups=c
            )
        else:
            lo = lo2 = smooth

        hi = img - lo2

        return lo, hi

    def forward(self, img):
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


class PointPyramid2d(nn.Module):
    def __init__(
        self, in_shape, outdims, scale_n, positive, bias, init_range, downsample, type, align_corners=True, **kwargs
    ):
        super().__init__()
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.positive = positive
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
        self.initialize()

    def initialize(self):
        self.grid.data.uniform_(-self.init_range, self.init_range)
        self.features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def group_sparsity(self, group_size):
        f = self.features.size(1)
        n = f // group_size
        ret = 0
        for chunk in range(0, f, group_size):
            ret = ret + (self.features[:, chunk : chunk + group_size, ...].pow(2).mean(1) + 1e-12).sqrt().mean() / n
        return ret

    def feature_l1(self, average=True):
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def forward(self, x, shift=None):
        if self.positive:
            positive(self.features)
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


class FullGaussian2d(nn.Module):
    """
    A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned.
    Args:
        in_shape (list, tuple): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]. Default: 0.1
        init_sigma (float): The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is
            'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the
            covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        gauss_type (str): Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        grid_mean_predictor (dict): Parameters for a predictor of the mean grid locations. Has to have a form like
                        {
                        'hidden_layers':0,
                        'hidden_features':20,
                        'final_tanh': False,
                        }
        shared_features (dict): Used when the feature vectors are shared (within readout between neurons) or between
                this readout and other readouts. Has to be a dictionary of the form
               {
                    'match_ids': (numpy.array),
                    'shared_features': torch.nn.Parameter or None
                }
                The match_ids are used to match things that should be shared within or across scans.
                If `shared_features` is None, this readout will create its own features. If it is set to
                a feature Parameter of another readout, it will replace the features of this readout. It will be
                access in increasing order of the sorted unique match_ids. For instance, if match_ids=[2,0,0,1],
                there should be 3 features in order [0,1,2]. When this readout creates features, it will do so in
                that order.
        shared_grid (dict): Like `shared_features`. Use dictionary like
               {
                    'match_ids': (numpy.array),
                    'shared_grid': torch.nn.Parameter or None
                }
                See documentation of `shared_features` for specification.
        source_grid (numpy.array):
                Source grid for the grid_mean_predictor.
                Needs to be of size neurons x grid_mean_predictor[input_dimensions]
        shared_transform (torch.nn.Parameter or None):
                This is only used if grid_mean_predictor is not None. If `shared_transform` is None, this readout will
                create its own mu_transform. If it is set to a mu_transform Parameter of another readout, it will replace
                 the mu_transform of this readout up to an additional unique bias.
        init_noise (float):
                Std of the normal distribution used to initialize the weights and biases.
        init_transform_scale (float):
                Only used if grid_mean_predictor is not None and shared_transform is None. Scale for the random
                orthogonal matrix that the mu_transform is initialized with.

    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        init_mu_range=0.1,
        init_sigma=1,
        batch_sample=True,
        align_corners=True,
        gauss_type="full",
        grid_mean_predictor=None,
        shared_features=None,
        shared_grid=None,
        source_grid=None,
        shared_transform=None,
        init_noise=1e-3,
        init_transform_scale=0.2,
        **kwargs,
    ):

        super().__init__()

        # determines whether the Gaussian is isotropic or not
        self.gauss_type = gauss_type

        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")

        # store statistics about the images and neurons
        self.in_shape = in_shape
        self.outdims = outdims
        self.init_noise = init_noise
        self.init_transform_scale = init_transform_scale
        # sample a different location per example
        self.batch_sample = batch_sample

        # position grid shape
        self.grid_shape = (1, outdims, 1, 2)

        # the grid can be predicted from another grid
        self._predicted_grid = False
        self._shared_grid = False
        self._original_grid = not self._predicted_grid

        if grid_mean_predictor is None and shared_grid is None:
            self._mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron
        elif grid_mean_predictor is not None and shared_grid is not None:
            raise ConfigurationError("Shared grid_mean_predictor and shared_grid_mean cannot both be set")
        elif grid_mean_predictor is not None:
            self.init_grid_predictor(source_grid=source_grid, shared_transform=shared_transform, **grid_mean_predictor)
        elif shared_grid is not None:
            self.initialize_shared_grid(**(shared_grid or {}))

        if gauss_type == "full":
            self.sigma_shape = (1, outdims, 2, 2)
        elif gauss_type == "uncorrelated":
            self.sigma_shape = (1, outdims, 1, 2)
        elif gauss_type == "isotropic":
            self.sigma_shape = (1, outdims, 1, 1)
        else:
            raise ValueError(f'gauss_type "{gauss_type}" not known')

        self.init_sigma = init_sigma
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))  # standard deviation for gaussian for each neuron

        self.initialize_features(**(shared_features or {}))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.init_mu_range = init_mu_range
        self.align_corners = align_corners
        self.initialize()

    @property
    def shared_features(self):
        return self._features

    @property
    def shared_grid(self):
        return self._mu

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[..., self.feature_sharing_index]
        else:
            return self._features

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    def feature_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if self._original_features:
            if average:
                return self._features.abs().mean()
            else:
                return self._features.abs().sum()
        else:
            return 0

    @property
    def mu(self):
        if self._predicted_grid:
            return self.mu_transform(self.source_grid.squeeze()).view(*self.grid_shape)
        elif self._shared_grid:
            if self._original_grid:
                return self._mu[:, self.grid_sharing_index, ...]
            else:
                return self.mu_transform(self._mu.squeeze())[self.grid_sharing_index].view(*self.grid_shape)
        else:
            return self._mu

    def sample_grid(self, batch_size, sample=None):
        """
        Returns the grid locations from the core by sampling from a Gaussian distribution
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """
        with torch.no_grad():
            self.mu.clamp_(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
            if self.gauss_type != "full":
                self.sigma.clamp_(min=0)  # sigma/variance i    s always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]

        sample = self.training if sample is None else sample
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

        if self.gauss_type != "full":
            return torch.clamp(
                norm * self.sigma + self.mu, min=-1, max=1
            )  # grid locations in feature space sampled randomly around the mean self.mu
        else:
            return torch.clamp(
                torch.einsum("ancd,bnid->bnic", self.sigma, norm) + self.mu, min=-1, max=1
            )  # grid locations in feature space sampled randomly around the mean self.mu

    def init_grid_predictor(
        self, source_grid, hidden_features=20, hidden_layers=0, final_tanh=False, shared_transform=None
    ):
        self._original_grid = False
        if shared_transform is None:
            layers = [nn.Linear(source_grid.shape[1], hidden_features if hidden_layers > 0 else 2)]

            for i in range(hidden_layers):
                layers.extend([nn.ELU(), nn.Linear(hidden_features, hidden_features if i < hidden_layers - 1 else 2)])

            if final_tanh:
                layers.append(nn.Tanh())
            self.mu_transform = nn.Sequential(*layers)
        else:
            self.mu_transform = BiasNet(base_net=shared_transform)
        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer("source_grid", torch.from_numpy(source_grid.astype(np.float32)))
        self._predicted_grid = True

    def initialize(self):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """

        if not self._predicted_grid or self._original_grid:
            self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.gauss_type != "full":
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)

        self._features.data.normal_(0, self.init_noise)

        if self._predicted_grid:
            if isinstance(self.mu_transform, nn.Sequential):
                for layer in self.mu_transform:
                    layer.bias.data.normal_(0, self.init_noise)
                if len(self.mu_transform) == 1:
                    self.mu_transform[0].weight.data = torch.from_numpy(
                        self.init_transform_scale * ortho_group.rvs(2).astype(np.float32)
                    )
            else:
                self.mu_transform.bias.data.normal_(0, self.init_noise)

        if self._shared_grid:
            self.mu_transform.bias.data.normal_(0, self.init_noise)
            self.mu_transform.weight.data = torch.eye(2)

        if self._shared_features:
            self.scales.data.normal_(1.0, self.init_noise)
        if self.bias is not None:
            self.bias.data.normal_(0, self.init_noise)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, w, h = self.in_shape
        self._original_features = True
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (
                    1,
                    c,
                    1,
                    n_match_ids,
                ), f"shared features need to have shape (1, {c}, 1, {n_match_ids})"
                self._features = shared_features
                self._original_features = False
            else:
                self._features = Parameter(
                    torch.Tensor(1, c, 1, n_match_ids)
                )  # feature weights for each channel of the core
            self.scales = Parameter(torch.Tensor(1, 1, 1, self.outdims))  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = Parameter(
                torch.Tensor(1, c, 1, self.outdims)
            )  # feature weights for each channel of the core
            self._shared_features = False

    def initialize_shared_grid(self, match_ids=None, shared_grid=None):
        c, w, h = self.in_shape

        if match_ids is None:
            raise ConfigurationError("match_ids must be set for sharing grid")
        assert self.outdims == len(match_ids), "There must be one match ID per output dimension"

        n_match_ids = len(np.unique(match_ids))
        if shared_grid is not None:
            assert shared_grid.shape == (
                1,
                n_match_ids,
                1,
                2,
            ), f"shared grid needs to have shape (1, {n_match_ids}, 1, 2)"
            self._mu = shared_grid
            self._original_grid = False
            self.mu_transform = nn.Linear(2, 2)
        else:
            self._mu = Parameter(torch.Tensor(1, n_match_ids, 1, 2))  # feature weights for each channel of the core
        _, sharing_idx = np.unique(match_ids, return_inverse=True)
        self.register_buffer("grid_sharing_index", torch.from_numpy(sharing_idx))
        self._shared_grid = True

    def forward(self, x, sample=None, shift=None, out_idx=None):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted
        Returns:
            y: neuronal activity
        """
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")
        feat = self.features.view(1, c, self.outdims)
        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, 2)

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = grid[:, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.gauss_type + " "
        r += self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        if self._shared_features:
            r += ", with {} features".format("original" if self._original_features else "shared")

        if self._predicted_grid:
            r += ", with predicted grid"
        if self._shared_grid:
            r += ", with {} grid".format("original" if self._original_grid else "shared")

        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r


class Gaussian3d(nn.Module):
    """
    This readout instantiates an object that can used to learn a point in the core feature space for each neuron,
    sampled from a Gaussian distribution with some mean and variance at train but set to mean at test time, that best predicts its response.

    The readout receives the shape of the core as 'in_shape', the number of units/neurons being predicted as 'outdims', 'bias' specifying whether
    or not bias term is to be used and 'init_range' range for initialising the mean and variance of the gaussian distribution from which we sample to
    uniform distribution, U(-init_mu_range,init_mu_range) and  uniform distribution, U(0.0, init_sigma_range) respectively.
    The grid parameter contains the normalized locations (x, y coordinates in the core feature space) and is clipped to [-1.1] as it a
    requirement of the torch.grid_sample function. The feature parameter learns the best linear mapping between the feature
    map from a given location, sample from Gaussian at train time but set to mean at eval time, and the unit's response with or without an additional elu non-linearity.

    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]
        init_sigma_range (float): initialises sigma with Uniform([0.0, init_sigma_range]).
                It is recommended however to use a fixed initialization, for faster convergence.
                For this, set fixed_sigma to True.
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        fixed_sigma (bool). Recommended behavior: True. But set to false for backwards compatibility.
                If true, initialized the sigma not in a range, but with the exact value given for all neurons.
    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        init_mu_range=0.5,
        init_sigma_range=0.5,
        batch_sample=True,
        align_corners=True,
        fixed_sigma=False,
        **kwargs,
    ):
        super().__init__()
        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma_range <= 0.0:
            raise ValueError("init_mu_range or init_sigma_range is not within required limit!")
        self.in_shape = in_shape
        self.outdims = outdims
        self.batch_sample = batch_sample
        self.grid_shape = (1, 1, outdims, 1, 3)
        self.mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron
        self.sigma = Parameter(torch.Tensor(*self.grid_shape))  # standard deviation for gaussian for each neuron
        self.features = Parameter(torch.Tensor(1, 1, 1, outdims))  # saliency weights for each channel from core

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.init_mu_range = init_mu_range
        self.init_sigma_range = init_sigma_range
        self.align_corners = align_corners
        self.fixed_sigma = fixed_sigma
        self.initialize()

    def sample_grid(self, batch_size, sample=None):
        """
        Returns the grid locations from the core by sampling from a Gaussian distribution
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """

        with torch.no_grad():
            self.mu.clamp_(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
            self.sigma.clamp_(min=0)  # sigma/variance is always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]

        sample = self.training if sample is None else sample

        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

        return torch.clamp(
            norm * self.sigma + self.mu, min=-1, max=1
        )  # grid locations in feature space sampled randomly around the mean self.mu

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    def initialize(self):
        self.mu.data.uniform_(-self.init_mu_range, self.init_mu_range)
        if self.fixed_sigma:
            self.sigma.data.uniform_(self.init_sigma_range, self.init_sigma_range)
        else:
            self.sigma.data.uniform_(0, self.init_sigma_range)
            warnings.warn(
                "sigma is sampled from uniform distribuiton, instead of a fixed value. Consider setting "
                "fixed_sigma to True"
            )
        self.features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, sample=None, shift=None, out_idx=None):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted

        Returns:
            y: neuronal activity
        """
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")
        x = x.view(N, 1, c, w, h)
        feat = self.features
        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, 3)

        if out_idx is not None:
            # out_idx specifies the indices to subset of neurons for training/testing
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, :, out_idx]
            grid = grid[:, :, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y


class UltraSparse(nn.Module):
    """
    This readout instantiates an object that can used to learn one or more features (with or without
    a shared mean in the x-y plane) in the core feature space for each neuron, sampled from a Gaussian distribution
    with some mean and variance at training but set to mean at test time, that best predicts its response.

    The readout receives the shape of the core as 'in_shape', the number of units/neurons being predicted as 'outdims', 'bias' specifying whether
    or not bias term is to be used and 'init_range' range for initialising the mean and variance of the gaussian distribution from which we sample to
    uniform distribution, U(-init_mu_range,init_mu_range) and  uniform distribution, U(0.0, init_sigma_range) respectively.
    The grid parameter contains the normalized locations (x, y coordinates in the core feature space) and is clipped to [-1.1] as it a
    requirement of the torch.grid_sample function. The feature parameter learns the best linear mapping between the feature
    map from a given location, sample from Gaussian at train time but set to mean at eval time, and the unit's response with or without an additional elu non-linearity.

    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]
        init_sigma_range (float): initialises sigma with Uniform([0.0, init_sigma_range])
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        num_filters (int): number of points in the core-features to be learned for each neuron
                           [default: 1, an instance of sparsest readout]
        shared_mean (bool): if True, the mean in the x-y plane (image-plane) is shared across all channels
                           [default: False]

        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        fixed_sigma (bool). Recommended behavior: True. But set to false for backwards compatibility.
                If true, initialized the sigma not in a range, but with the exact value given for all neurons.
    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        init_mu_range,
        init_sigma_range,
        batch_sample=True,
        num_filters=1,
        shared_mean=False,
        align_corners=True,
        fixed_sigma=False,
        **kwargs,
    ):

        super().__init__()
        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma_range <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive!")
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.batch_sample = batch_sample
        self.num_filters = num_filters
        self.shared_mean = shared_mean
        self.grid_shape = (1, 1, outdims * num_filters, 1, 3)

        if shared_mean:

            self.gridxy_shape = (1, 1, outdims, 1, 2)
            self.gridch_shape = (1, 1, outdims * num_filters, 1, 1)
            self.mu_xy = Parameter(
                torch.Tensor(*self.gridxy_shape)
            )  # mean location (in xy dim) of gaussian for each neuron
            self.mu_ch = Parameter(
                torch.Tensor(*self.gridch_shape)
            )  # mean location (in ch dim) of gaussian for each neuron
            self.sigma_xy = Parameter(
                torch.Tensor(*self.gridxy_shape)
            )  # standard deviation for gaussian for each neuron
            self.sigma_ch = Parameter(torch.Tensor(*self.gridch_shape))

        else:

            self.mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron
            self.sigma = Parameter(torch.Tensor(*self.grid_shape))  # standard deviation for gaussian for each neuron

        self.features = Parameter(
            torch.Tensor(1, 1, outdims, num_filters)
        )  # saliency  weights for each channel from core

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.init_mu_range = init_mu_range
        self.init_sigma_range = init_sigma_range
        self.align_corners = align_corners
        self.fixed_sigma = fixed_sigma
        self.initialize()

    def sample_grid(self, batch_size, sample=None):
        """
        Returns the grid locations from the core by sampling from a Gaussian distribution
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """

        if self.shared_mean:
            # sample an xy location and keep it same across all filter channels
            # explicit clamping of mu and sigma along the channel dimension was needed as the clamping post cat was not working
            with torch.no_grad():
                self.mu_ch.clamp_(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
                self.sigma_ch.clamp_(min=0)  # sigma/variance is always a positive quantity
            self.mu = torch.cat((self.mu_xy.repeat(1, 1, self.num_filters, 1, 1), self.mu_ch), 4)
            self.sigma = torch.cat((self.sigma_xy.repeat(1, 1, self.num_filters, 1, 1), self.sigma_ch), 4)

        with torch.no_grad():
            self.mu.clamp_(min=-1, max=1)
            self.sigma.clamp_(min=0)

        grid_shape = (batch_size,) + self.grid_shape[1:]

        sample = self.training if sample is None else sample

        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

        return torch.clamp(
            norm * self.sigma + self.mu, min=-1, max=1
        )  # grid locations in feature space sampled randomly around the mean self.mu

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    def feature_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def initialize(self):

        if self.shared_mean:
            # initialise mu and sigma separately for xy and channel dimension.
            self.mu_ch.data.uniform_(-1, 1)
            self.mu_xy.data.uniform_(-self.init_mu_range, self.init_mu_range)

            if self.fixed_sigma:
                self.sigma_ch.data.uniform_(self.init_sigma_range, self.init_sigma_range)
                self.sigma_xy.data.uniform_(self.init_sigma_range, self.init_sigma_range)
            else:
                self.sigma_ch.data.uniform_(0, self.init_sigma_range)
                self.sigma_xy.data.uniform_(0, self.init_sigma_range)
                warnings.warn(
                    "sigma is sampled from uniform distribuiton, instead of a fixed value. Consider setting "
                    "fixed_sigma to True"
                )

        else:
            # initialise mu and sigma for x,y and channel dimensions.
            self.mu.data.uniform_(-self.init_mu_range, self.init_mu_range)
            self.sigma.data.uniform_(0, self.init_sigma_range)

        self.features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, sample=True, shift=None, out_idx=None):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted

        Returns:
            y: neuronal activity
        """

        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")
        x = x.view(N, 1, c, w, h)
        feat = self.features
        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(N, 1, outdims * self.num_filters, 1, 3)

        if out_idx is not None:
            # predict output only for neurons given by out_idx
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, :, out_idx]
            grid = grid[:, :, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:  # it might not be valid now but have kept it for future devop.
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners).squeeze(-1)
        z = y.view((N, 1, self.num_filters, outdims)).permute(0, 1, 3, 2)  # reorder the dims
        z = torch.einsum(
            "nkpf,mkpf->np", z, feat
        )  # dim: batch_size, 1, num_neurons, num_filters -> batch_size, num_neurons

        if self.bias is not None:
            z = z + bias
        return z

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r


# ------------ Multi Readouts ------------------------


class MultiReadout(Readout, ModuleDict):
    _base_readout = None

    def __init__(self, in_shape, loaders, gamma_readout, clone_readout=False, **kwargs):
        if self._base_readout is None:
            raise ValueError("Attribute _base_readout must be set")

        super().__init__()

        self.in_shape = in_shape
        self.neurons = OrderedDict([(k, loader.dataset.n_neurons) for k, loader in loaders.items()])
        if "positive" in kwargs:
            self._positive = kwargs["positive"]

        self.gamma_readout = gamma_readout  # regularisation strength

        for i, (k, n_neurons) in enumerate(self.neurons.items()):
            if i == 0 or clone_readout is False:
                self.add_module(k, self._base_readout(in_shape=in_shape, outdims=n_neurons, **kwargs))
                original_readout = k
            elif i > 0 and clone_readout is True:
                self.add_module(k, ClonedReadout(self[original_readout], **kwargs))

    def initialize(self, mean_activity_dict):
        for k, mu in mean_activity_dict.items():
            self[k].initialize()
            if hasattr(self[k], "bias"):
                self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].feature_l1() * self.gamma_readout

    @property
    def positive(self):
        if hasattr(self, "_positive"):
            return self._positive
        else:
            return False

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value


class MultiplePointPyramid2d(MultiReadout):
    _base_readout = PointPyramid2d


class MultipleGaussian3d(MultiReadout):
    """
    Instantiates multiple instances of Gaussian3d Readouts
    usually used when dealing with different datasets or areas sharing the same core.
    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        loaders (list):  a list of dataset objects
        gamma_readout (float): regularisation term for the readout which is usally set to 0.0 for gaussian3d readout
                               as it contains one dimensional weight

    """

    _base_readout = Gaussian3d

    # Make sure this is not a bug
    def regularizer(self, readout_key):
        return self.gamma_readout


class MultiplePointPooled2d(MultiReadout):
    """
    Instantiates multiple instances of PointPool2d Readouts
    usually used when dealing with more than one dataset sharing the same core.
    """

    _base_readout = PointPooled2d


class MultipleFullGaussian2d(MultiReadout):
    """
    Instantiates multiple instances of FullGaussian2d Readouts
    usually used when dealing with more than one dataset sharing the same core.

    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        loaders (list):  a list of dataloaders
        gamma_readout (float): regularizer for the readout
    """

    _base_readout = FullGaussian2d


class MultipleUltraSparse(MultiReadout):
    """
    This class instantiates multiple instances of UltraSparseReadout
    useful when dealing with multiple datasets
    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        loaders (list):  a list of dataset objects
        gamma_readout (float): regularisation term for the readout which is usally set to 0.0 for UltraSparseReadout readout
                               as it contains one dimensional weight
    """

    _base_readout = UltraSparse
