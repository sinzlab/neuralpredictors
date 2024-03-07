import itertools
import math
from collections import OrderedDict

import torch
from torch import nn

from neuralpredictors import regularizers
from neuralpredictors.layers.activations import AdaptiveELU
from neuralpredictors.utils import check_hyperparam_for_layers

from ...regularizers import DepthLaplaceL21d
from ..affine import Bias3DLayer, Scale3DLayer
from .base import Core


class Core3d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)
        self.put_to_cuda(cuda=cuda)

    def put_to_cuda(self, cuda):
        if cuda:
            self.cuda()

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)


class Basic3dCore(Core3d, nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kernel,
        hidden_kernel,
        layers=3,
        stride=1,
        gamma_input_spatial=0,
        gamma_input_temporal=0,
        hidden_nonlinearities="elu",
        x_shift=0,
        y_shift=0,
        bias=True,
        batch_norm=True,
        padding=False,
        batch_norm_scale=True,
        momentum=0.01,
        laplace_padding=None,
        input_regularizer="LaplaceL2norm",
        cuda=False,
        final_nonlin=True,
        independent_bn_bias=True,
        spatial_dilation: int = 1,
        temporal_dilation: int = 1,
        hidden_spatial_dilation=1,
        hidden_temporal_dilation=1,
    ):

        """
        :param input_channels: integer, number of input channels as in
        :param hidden_channels:  number of hidden channels (i.e feature maps) in each hidden layer
        :param input_kernel: kernel size of the first layer (i.e. the input layer)
        :param hidden_kernel: kernel size of each hidden layer's kernel
        :param layers: number of layers
        :param stride: the stride of the convolutions
        :param spatial_dilation: dilation of ONLY the first spatial kernel (both width and height)
        :param temporal dilation: dilation of ONLY the first temporal kernel
        :param gamma_input_spatial: regularizer factor for spatial smoothing
        :param gamma_input_temporal: regularizer factor for temporal smoothing
        :param hidden_nonlinearities: what kind of non-linearity is applied in core, should be one of self.nonlinearities
        :param x_shift: shift in x axis in case ELU is the nonlinearity
        :param y_shift: shift in y axis in case ELU is the nonlinearity
        :param bias: adds a bias layer
        :param batch_norm: bool specifying whether to include batch norm after convolution in core
        :param padding: whether to pad convolutions. Defaults to False
        :param batch_norm_scale: bool, if True, a scaling factor after BN will be learned.
        :param momentum: momentum for batch norm
        :param laplace_padding: padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
        :param input_regularizer: specifies what kind of spatial regularized is applied
        :param final_nonlin: bool specifiyng whether to include a nonlinearity after last convolutional layer in core
        :param independent_bn_bias: If False, will allow for scaling the batch norm, so that batch norm
                                    and bias can both be true. Defaults to True.

        To enable learning batch_norms bias and scale independently, the arguments bias, batch_norm and batch_norm_scale
        work together: By default, all are true. In this case there won't be a bias learned in the convolutional layer, but
        batch_norm will learn both its bias and scale. If batch_norm is false, but bias true, a bias will be learned in the
        convolutional layer. If batch_norm and bias are true, but batch_norm_scale is false, batch_norm won't have learnable
        parameters and a BiasLayer will be added after the batch_norm layer.
        """
        super().__init__()

        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kernel)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weight_regularizer = getattr(regularizers, input_regularizer)(**regularizer_config)
        self.temporal_regularizer = DepthLaplaceL21d()
        self.layers = layers
        self.input_channels = input_channels
        self.input_kernel = input_kernel
        self.hidden_channels = hidden_channels
        self.hidden_kernel = hidden_kernel
        self.stride = stride
        self.bias = bias
        self.batch_norm = batch_norm
        self.batch_norm_scale = batch_norm_scale
        self.independent_bn_bias = independent_bn_bias
        self.momentum = momentum
        self.spatial_dilation = spatial_dilation
        self.temporal_dilation = temporal_dilation
        self.hidden_spatial_dilation = spatial_dilation
        self.hidden_temporal_dilation = temporal_dilation
        self.final_nonlinearity = final_nonlin
        self.padding = padding
        self.gamma_input_spatial = gamma_input_spatial
        self.gamma_input_temporal = gamma_input_temporal
        self.nonlinearities = {
            "elu": torch.nn.ELU,
            "softplus": torch.nn.Softplus,
            "relu": torch.nn.ReLU,
            "adaptive_elu": AdaptiveELU,
        }

        if isinstance(self.hidden_channels, int):
            if self.layers >= 1:
                self.hidden_channels = [hidden_channels] * (self.layers)
            else:
                self.hidden_channels = []

        self.hidden_channels = check_hyperparam_for_layers(hidden_channels, layers)
        self.hidden_temporal_dilation = check_hyperparam_for_layers(hidden_temporal_dilation, layers)
        self.hidden_spatial_dilation = check_hyperparam_for_layers(hidden_spatial_dilation, layers)

        if isinstance(self.input_kernel, int):
            self.input_kernel = (self.input_kernel,) * 3

        if isinstance(self.hidden_kernel, int):
            self.hidden_kernel = (self.hidden_kernel,) * 3

        if isinstance(self.hidden_kernel, (tuple, list)):
            if self.layers > 1:
                self.hidden_kernel = [self.hidden_kernel] * (self.layers - 1)
            else:
                self.hidden_kernel = []

        if isinstance(self.stride, int):
            self.stride = [self.stride] * self.layers

        self.features = nn.Sequential()
        layer = OrderedDict()
        layer["conv"] = nn.Conv3d(
            in_channels=input_channels,
            out_channels=self.hidden_channels[0],
            kernel_size=input_kernel,
            stride=(1, self.stride[0], self.stride[0]),
            dilation=(self.temporal_dilation, self.spatial_dilation, self.spatial_dilation),
            bias=self.bias,
            padding=(0, input_kernel[1] // 2, input_kernel[2] // 2) if self.padding else 0,
        )

        self.add_bn_layer(layer=layer, hidden_channels=self.hidden_channels[0])

        if layers > 1 or self.final_nonlinearity:
            if hidden_nonlinearities == "adaptive_elu":
                layer["nonlin"] = self.nonlinearities[hidden_nonlinearities](xshift=x_shift, yshift=y_shift)
            else:
                layer["nonlin"] = self.nonlinearities[hidden_nonlinearities]()

        self.features.add_module("layer0", nn.Sequential(layer))

        for l in range(0, self.layers - 1):
            layer = OrderedDict()
            layer[f"conv_{l + 1}"] = nn.Conv3d(
                self.hidden_channels[l],
                self.hidden_channels[l + 1],
                kernel_size=self.hidden_kernel[l],
                dilation=(
                    self.hidden_temporal_dilation[l],
                    self.hidden_spatial_dilation[l],
                    self.hidden_spatial_dilation[l],
                ),
                stride=(1, self.stride[l + 1], self.stride[l + 1]),
                padding=(0, self.hidden_kernel[l][1] // 2, self.hidden_kernel[l][2] // 2) if self.padding else 0,
            )

            self.add_bn_layer(layer=layer, hidden_channels=self.hidden_channels[l + 1])

            if self.final_nonlinearity or l < self.layers:
                if hidden_nonlinearities == "adaptive_elu":
                    layer["nonlin"] = self.nonlinearities[hidden_nonlinearities](x_shift=x_shift, y_shift=y_shift)
                else:
                    layer["nonlin"] = self.nonlinearities[hidden_nonlinearities]()

            self.features.add_module("layer{}".format(l + 1), nn.Sequential(layer))

        self.initialize(cuda=cuda)

    def forward(self, x):
        for features in self.features:
            x = features(x)
        return x

    def laplace_spatial(self):
        laplace = 0
        for filter_index in range(self.features[0].conv.weight.shape[2]):
            laplace += self._input_weight_regularizer(self.features[0].conv.weight[:, :, filter_index, :, :])
        return laplace

    def laplace_temporal(self):
        laplace = 0
        for w, h in itertools.product(
            range(self.features[0].conv.weight.shape[-2]), range(self.features[0].conv.weight.shape[-1])
        ):
            laplace += self.temporal_regularizer(self.features[0].conv.weight[:, :, :, w, h])
        return laplace

    def regularizer(self):
        return self.gamma_input_spatial * self.laplace_spatial(), self.gamma_input_temporal * self.laplace_temporal()

    def add_bn_layer(self, layer, hidden_channels):
        if self.batch_norm:
            if self.independent_bn_bias:
                layer["norm"] = nn.BatchNorm3d(hidden_channels, momentum=self.momentum)
            else:
                layer["norm"] = nn.BatchNorm3d(
                    hidden_channels, momentum=self.momentum, affine=self.bias and self.batch_norm_scale
                )
                if self.bias and not self.batch_norm_scale:
                    layer["bias"] = Bias3DLayer(hidden_channels)
                elif self.batch_norm_scale:
                    layer["scale"] = Scale3DLayer(hidden_channels)

    @property
    def out_channels(self):
        return self.hidden_channels[-1]

    def get_kernels(self):
        return [self.input_kernel] + [kernel for kernel in self.hidden_kernel]


class Factorized3dCore(Core3d, nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        spatial_input_kernel,
        temporal_input_kernel,
        spatial_hidden_kernel,
        temporal_hidden_kernel,
        final_nonlin,
        layers=3,
        stride=1,
        x_shift=0.0,
        y_shift=0.0,
        gamma_input_spatial=0,
        gamma_input_temporal=0,
        hidden_nonlinearities="elu",
        bias=True,
        batch_norm=True,
        padding=False,
        batch_norm_scale=True,
        independent_bn_bias=True,
        momentum=0.01,
        laplace_padding=None,
        input_regularizer="LaplaceL2norm",
        cuda=False,
        spatial_dilation=1,
        temporal_dilation=1,
        hidden_spatial_dilation=1,
        hidden_temporal_dilation=1,
    ):
        """
        Core3d, similar to Basic3dCore but the convolution is separated into first spatial and then temporal.

        :param input_channels: integer, number of input channels as in
        :param hidden_channels: number of hidden channels (i.e feature maps) in each hidden layer
        :param spatial_input_kernel: kernel size of the first spatial layer (i.e. the input layer)
        :param temporal_input_kernel: kernel size of the temporal layer
        :param spatial_hidden_kernel:  kernel size of each hidden layer's spatial kernel
        :param temporal_hidden_kernel:  kernel size of each hidden layer's temporal kernel
        :param spatial_dilation: dilation of ONLY the first spatial kernel (both width and height)
        :param temporal dilation: dilation of ONLY the first temporal kernel
        :param final_nonlin: bool specifiyng whether to include a nonlinearity after last core convolution
        :param layers: number of layers
        :param stride: the stride of the convolutions.
        :param x_shift: shift in x axis in case ELU is the nonlinearity
        :param y_shift: shift in y axis in case ELU is the nonlinearity
        :param gamma_input_spatial: regularizer factor for spatial smoothing
        :param gamma_input_temporal: regularizer factor for temporal smoothing
        :param hidden_nonlinearities:
        :param bias: adds a bias layer - TODO: actually now does not do anything I think
        :param batch_norm: bool specifying whether to include batch norm after convolution in core
        :param padding: whether to pad convolutions. Defaults to False.
        :param batch_norm_scale: bool, if True, a scaling factor after BN will be learned.
        :param independent_bn_bias: If False, will allow for scaling the batch norm, so that batchnorm
                                    and bias can both be true. Defaults to True.
        :param momentum: momentum for batch norm
        :param laplace_padding: padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
        :param input_regularizer: specifies what kind of spatial regularized is applied. Must match one of the
                                  regularizers in neuralpredictors.regularizers
        :param cuda:
        """
        super().__init__()

        regularizer_config = (
            dict(padding=laplace_padding, kernel=spatial_input_kernel)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weight_regularizer = getattr(regularizers, input_regularizer)(**regularizer_config)
        self.temporal_regularizer = DepthLaplaceL21d()
        self.layers = layers
        self.input_channels = input_channels
        self.spatial_input_kernel = spatial_input_kernel
        self.temporal_input_kernel = temporal_input_kernel
        self.hidden_channels = hidden_channels
        self.spatial_hidden_kernel = spatial_hidden_kernel
        self.temporal_hidden_kernel = temporal_hidden_kernel
        self.bias = bias
        self.batch_norm = batch_norm
        self.batch_norm_scale = batch_norm_scale
        self.independent_bn_bias = independent_bn_bias
        self.momentum = momentum
        self.stride = stride
        self.spatial_dilation = spatial_dilation
        self.temporal_dilation = temporal_dilation
        self.hidden_spatial_dilation = (hidden_spatial_dilation,)
        self.hidden_temporal_dilation = (hidden_temporal_dilation,)
        self.padding = padding
        self.gamma_input_spatial = gamma_input_spatial
        self.gamma_input_temporal = gamma_input_temporal
        self.nonlinearities = {
            "elu": torch.nn.ELU,
            "softplus": torch.nn.Softplus,
            "relu": torch.nn.ReLU,
            "adaptive_elu": AdaptiveELU,
        }

        self.hidden_channels = check_hyperparam_for_layers(hidden_channels, self.layers)
        self.hidden_temporal_dilation = check_hyperparam_for_layers(hidden_temporal_dilation, self.layers - 1)
        self.hidden_spatial_dilation = check_hyperparam_for_layers(hidden_spatial_dilation, self.layers - 1)

        if isinstance(self.spatial_input_kernel, int):
            self.spatial_input_kernel = (self.spatial_input_kernel,) * 2

        if isinstance(self.spatial_hidden_kernel, int):
            self.spatial_hidden_kernel = (self.spatial_hidden_kernel,) * 2

        if isinstance(self.spatial_hidden_kernel, (tuple, list)):
            if self.layers > 1:
                self.spatial_hidden_kernel = [self.spatial_hidden_kernel] * (self.layers - 1)
                self.temporal_hidden_kernel = [self.temporal_hidden_kernel] * (self.layers - 1)
            else:
                self.spatial_hidden_kernel = []
                self.temporal_hidden_kernel = []

        if isinstance(self.stride, int):
            self.stride = [self.stride] * self.layers

        self.features = nn.Sequential()
        layer = OrderedDict()
        layer["conv_spatial"] = nn.Conv3d(
            in_channels=input_channels,
            out_channels=self.hidden_channels[0],
            kernel_size=(1,) + self.spatial_input_kernel,
            stride=(1, self.stride[0], self.stride[0]),
            bias=self.bias,
            dilation=(1, self.spatial_dilation, self.spatial_dilation),
            padding=(0, self.spatial_input_kernel[0] // 2, self.spatial_input_kernel[1] // 2) if self.padding else 0,
        )
        layer["conv_temporal"] = nn.Conv3d(
            self.hidden_channels[0],
            self.hidden_channels[0],
            kernel_size=(temporal_input_kernel, 1, 1),
            bias=self.bias,
            dilation=(self.temporal_dilation, 1, 1),
        )

        self.add_bn_layer(layer=layer, hidden_channels=self.hidden_channels[0])

        if layers > 1 or final_nonlin:
            if hidden_nonlinearities == "adaptive_elu":
                layer["nonlin"] = self.nonlinearities[hidden_nonlinearities](xshift=x_shift, yshift=y_shift)
            else:
                layer["nonlin"] = self.nonlinearities[hidden_nonlinearities]()

        self.features.add_module("layer0", nn.Sequential(layer))

        for l in range(0, self.layers - 1):
            layer = OrderedDict()
            layer[f"conv_spatial_{l+1}"] = nn.Conv3d(
                in_channels=self.hidden_channels[l],
                out_channels=self.hidden_channels[l + 1],
                kernel_size=(1,) + (self.spatial_hidden_kernel[l]),
                stride=(1, self.stride[l], self.stride[l]),
                bias=self.bias,
                dilation=(1, self.hidden_spatial_dilation[l], self.hidden_spatial_dilation[l]),
                padding=(0, self.spatial_hidden_kernel[l][0] // 2, self.spatial_hidden_kernel[l][1] // 2)
                if self.padding
                else 0,
            )
            layer[f"conv_temporal_{l+1}"] = nn.Conv3d(
                self.hidden_channels[l + 1],
                self.hidden_channels[l + 1],
                kernel_size=(self.temporal_hidden_kernel[l], 1, 1),
                bias=self.bias,
                dilation=(self.hidden_temporal_dilation[l], 1, 1),
            )

            self.add_bn_layer(layer=layer, hidden_channels=self.hidden_channels[l + 1])

            if final_nonlin or l < self.layers:
                if hidden_nonlinearities == "adaptive_elu":
                    layer["nonlin"] = self.nonlinearities[hidden_nonlinearities](x_shift=x_shift, y_shift=y_shift)
                else:
                    layer["nonlin"] = self.nonlinearities[hidden_nonlinearities]()

            self.features.add_module("layer{}".format(l + 1), nn.Sequential(layer))
        self.initialize(cuda=cuda)

    def forward(self, x):
        for features in self.features:
            x = features(x)
        return x

    def laplace_spatial(self):
        laplace = 0
        laplace += self._input_weight_regularizer(self.features[0].conv_spatial.weight[:, :, 0, :, :])
        return laplace

    def laplace_temporal(self):
        laplace = self.temporal_regularizer(self.features[0].conv_temporal.weight[:, :, :, 0, 0])
        return laplace

    def regularizer(self):
        return self.gamma_input_spatial * self.laplace_spatial(), self.gamma_input_temporal * self.laplace_temporal()

    def get_kernels(self):
        return [(self.temporal_input_kernel,) + self.spatial_input_kernel] + [
            (temporal_kernel,) + spatial_kernel
            for temporal_kernel, spatial_kernel in zip(self.temporal_hidden_kernel, self.spatial_hidden_kernel)
        ]

    def add_bn_layer(self, layer, hidden_channels):
        if self.batch_norm:
            if self.independent_bn_bias:
                layer["norm"] = nn.BatchNorm3d(hidden_channels, momentum=self.momentum)
            else:
                layer["norm"] = nn.BatchNorm3d(
                    hidden_channels, momentum=self.momentum, affine=self.bias and self.batch_norm_scale
                )
                if self.bias and not self.batch_norm_scale:
                    layer["bias"] = Bias3DLayer(hidden_channels)
                elif self.batch_norm_scale:
                    layer["scale"] = Scale3DLayer(hidden_channels)
