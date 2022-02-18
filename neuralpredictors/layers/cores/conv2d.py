import logging
import warnings
from collections import Iterable, OrderedDict
from functools import partial

import torch
import torchvision
from torch import nn

from ... import regularizers
from ..activations import AdaptiveELU
from ..affine import Bias2DLayer, Scale2DLayer
from ..attention import AttentionConv
from ..conv import DepthSeparableConv2d
from ..hermite import (
    HermiteConv2D,
    RotationEquivariantBatchNorm2D,
    RotationEquivariantBias2DLayer,
    RotationEquivariantScale2DLayer,
)
from ..squeeze_excitation import SqueezeExcitationBlock
from .base import Core

logger = logging.getLogger(__name__)


class Stacked2dCore(Core, nn.Module):
    """
    An instantiation of the Core base class. Made up of layers layers of nn.sequential modules.
    Allows for the flexible implementations of many different architectures, such as convolutional layers,
    or self-attention layers.
    """

    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        layers=3,
        gamma_hidden=0,
        gamma_input=0.0,
        skip=0,
        stride=1,
        final_nonlinearity=True,
        elu_shift=(0, 0),
        bias=True,
        momentum=0.1,
        pad_input=True,
        hidden_padding=None,
        batch_norm=True,
        batch_norm_scale=True,
        independent_bn_bias=True,
        hidden_dilation=1,
        laplace_padding=0,
        input_regularizer="LaplaceL2",
        stack=None,
        use_avg_reg=True,
        depth_separable=False,
        attention_conv=False,
        linear=False,
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            gamma_hidden:   regularizer factor for group sparsity
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see neuralpredictors.regularizers)
            skip:           Adds a skip connection
            stride:         stride of the 2d conv layer.
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            elu_shift: a tuple to shift the elu in the following way: Elu(x - elu_xshift) + elu_yshift
            bias:           Adds a bias layer.
            momentum:       momentum in the batchnorm layer.
            pad_input:      Boolean, if True, applies zero padding to all convolutions
            hidden_padding: int or list of int. Padding for hidden layers. Note that this will apply to all the layers
                            except the first (input) layer.
            batch_norm:     Boolean, if True appends a BN layer after each convolutional layer
            batch_norm_scale: If True, a scaling factor after BN will be learned.
            independent_bn_bias:    If False, will allow for scaling the batch norm, so that batchnorm
                                    and bias can both be true. Defaults to True.
            hidden_dilation:    If set to > 1, will apply dilated convs for all hidden layers
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            input_regularizer:  String that must match one of the regularizers in ..regularizers
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            Implemented as layers_to_stack = layers[stack:]. thus:
                                stack = -1 will only select the last layer as the readout layer.
                                stack of -2 will read out from the last two layers.
                                And stack of 1 will read out from layer 1 (0 indexed) until the last layer.
            use_avg_reg:    bool. Whether to use the averaged value of regularizer(s) or the summed.
            depth_separable: Boolean, if True, uses depth-separable convolutions in all layers after the first one.
            attention_conv: Boolean, if True, uses self-attention instead of convolution for all layers after the first one.
            linear:         Boolean, if True, removes all nonlinearities

            To enable learning batch_norms bias and scale independently, the arguments bias, batch_norm and batch_norm_scale
            work together: By default, all are true. In this case there won't be a bias learned in the convolutional layer, but
            batch_norm will learn both its bias and scale. If batch_norm is false, but bias true, a bias will be learned in the
            convolutional layer. If batch_norm and bias are true, but batch_norm_scale is false, batch_norm won't have learnable
            parameters and a BiasLayer will be added after the batch_norm layer.
        """

        if depth_separable and attention_conv:
            raise ValueError("depth_separable and attention_conv can not both be true")

        super().__init__()
        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kern)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)
        self.num_layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.skip = skip
        self.stride = stride
        self.use_avg_reg = use_avg_reg
        if use_avg_reg:
            warnings.warn("The averaged value of regularizer will be used.", UserWarning)
        self.hidden_padding = hidden_padding
        self.input_kern = input_kern
        self.hidden_kern = hidden_kern
        self.laplace_padding = laplace_padding
        self.hidden_dilation = hidden_dilation
        self.final_nonlinearity = final_nonlinearity
        self.elu_xshift, self.elu_yshift = elu_shift
        self.bias = bias
        self.momentum = momentum
        self.pad_input = pad_input
        self.batch_norm = batch_norm
        self.batch_norm_scale = batch_norm_scale
        self.independent_bn_bias = independent_bn_bias
        if stack is None:
            self.stack = range(self.num_layers)
        else:
            self.stack = [*range(self.num_layers)[stack:]] if isinstance(stack, int) else stack
        self.linear = linear

        if depth_separable:
            self.conv_layer_name = "ds_conv"
            self.ConvLayer = DepthSeparableConv2d
            self.ignore_group_sparsity = True
        elif attention_conv:
            # TODO: check if name attention_conv is backwards compatible
            self.conv_layer_name = "attention_conv"
            self.ConvLayer = self.AttentionConvWrapper
            self.ignore_group_sparsity = True
        else:
            self.conv_layer_name = "conv"
            self.ConvLayer = nn.Conv2d
            self.ignore_group_sparsity = False

        if (self.ignore_group_sparsity) and (gamma_hidden > 0):
            warnings.warn(
                "group sparsity can not be calculated for the requested conv type. Hidden channels will not be regularized and gamma_hidden is ignored."
            )
        self.set_batchnorm_type()
        self.features = nn.Sequential()
        self.add_first_layer()
        self.add_subsequent_layers()
        self.initialize()

    def set_batchnorm_type(self):
        self.batchnorm_layer_cls = nn.BatchNorm2d
        self.bias_layer_cls = Bias2DLayer
        self.scale_layer_cls = Scale2DLayer

    def add_bn_layer(self, layer):
        if self.batch_norm:
            if self.independent_bn_bias:
                layer["norm"] = self.batchnorm_layer_cls(self.hidden_channels, momentum=self.momentum)
            else:
                layer["norm"] = self.batchnorm_layer_cls(
                    self.hidden_channels, momentum=self.momentum, affine=self.bias and self.batch_norm_scale
                )
                if self.bias:
                    if not self.batch_norm_scale:
                        layer["bias"] = self.bias_layer_cls(self.hidden_channels)
                elif self.batch_norm_scale:
                    layer["scale"] = self.scale_layer_cls(self.hidden_channels)

    def add_activation(self, layer):
        if self.linear:
            return
        if len(self.features) < self.num_layers - 1 or self.final_nonlinearity:
            layer["nonlin"] = AdaptiveELU(self.elu_xshift, self.elu_yshift)

    def add_first_layer(self):
        layer = OrderedDict()
        layer["conv"] = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.input_kern,
            padding=self.input_kern // 2 if self.pad_input else 0,
            bias=self.bias and not self.batch_norm,
        )
        self.add_bn_layer(layer)
        self.add_activation(layer)
        self.features.add_module("layer0", nn.Sequential(layer))

    def add_subsequent_layers(self):
        if not isinstance(self.hidden_kern, Iterable):
            self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

        for l in range(1, self.num_layers):
            layer = OrderedDict()
            if self.hidden_padding is None:
                self.hidden_padding = ((self.hidden_kern[l - 1] - 1) * self.hidden_dilation + 1) // 2
            layer[self.conv_layer_name] = self.ConvLayer(
                in_channels=self.hidden_channels if not self.skip > 1 else min(self.skip, l) * self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.hidden_kern[l - 1],
                stride=self.stride,
                padding=self.hidden_padding,
                dilation=self.hidden_dilation,
                bias=self.bias,
            )
            self.add_bn_layer(layer)
            self.add_activation(layer)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

    class AttentionConvWrapper(AttentionConv):
        def __init__(self, dilation=None, **kwargs):
            """
            Helper class to make an attention conv layer accept input args of a pytorch.nn.Conv2d layer.
            Args:
                dilation: catches this argument from the input args, and ignores it
                **kwargs:
            """
            super().__init__(**kwargs)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l) :], dim=1))
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def laplace(self):
        """
        Laplace regularization for the filters of the first conv2d layer.
        """
        return self._input_weights_regularizer(self.features[0].conv.weight, avg=self.use_avg_reg)

    def group_sparsity(self):
        """
        Sparsity regularization on the filters of all the conv2d layers except the first one.
        """
        ret = 0
        if self.ignore_group_sparsity:
            return ret

        for feature in self.features[1:]:
            ret = ret + feature.conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.num_layers - 1) if self.num_layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels


class RotationEquivariant2dCore(Stacked2dCore, nn.Module):
    """
    A core built of 2d rotation-equivariant layers. For more info refer to https://openreview.net/forum?id=H1fU8iAqKX.
    """

    def __init__(
        self,
        *args,
        num_rotations=8,
        stride=1,
        upsampling=2,
        rot_eq_batch_norm=True,
        input_regularizer="LaplaceL2norm",
        **kwargs,
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            num_rotations:  number of computed rotations for every feature
            stride:         stride in convolutional layers
            upsampling:     upsampling scale of Hermite filters
            gamma_hidden:   regularizer factor for group sparsity
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see neuralpredictors.regularizers)
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            elu_xshift, elu_yshift: final_nonlinearity(x) = Elu(x - elu_xshift) + elu_yshift
            bias:           Adds a bias layer.
            momentum:        momentum in the batchnorm layer.
            pad_input:      Boolean, if True, applies zero padding to all convolutions
            hidden_padding: int or list of int. Padding for hidden layers. Note that this will apply to all the layers
                            except the first (input) layer.
            batch_norm:     Boolean, if True appends a BN layer after each convolutional layer
            batch_norm_scale: If True, a scaling factor after BN will be learned.
            independent_bn_bias:    If False, will allow for scaling the batch norm, so that batchnorm
                                    and bias can both be true. Defaults to True.
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            input_regularizer:  String that must match one of the regularizers in ..regularizers
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            Implemented as layers_to_stack = layers[stack:]. thus:
                                stack = -1 will only select the last layer as the readout layer.
                                stack of -2 will read out from the last two layers.
                                And stack of 1 will read out from layer 1 (0 indexed) until the last layer.

            use_avg_reg:    bool. Whether to use the averaged value of regularizer(s) or the summed.

            To enable learning batch_norms bias and scale independently, the arguments bias, batch_norm and batch_norm_scale
            work together: By default, all are true. In this case there won't be a bias learned in the convolutional layer, but
            batch_norm will learn both its bias and scale. If batch_norm is false, but bias true, a bias will be learned in the
            convolutional layer. If batch_norm and bias are true, but batch_norm_scale is false, batch_norm won't have learnable
            parameters and a BiasLayer will be added after the batch_norm layer.
        """
        self.num_rotations = num_rotations
        self.stride = stride
        self.upsampling = upsampling
        self.rot_eq_batch_norm = rot_eq_batch_norm
        super().__init__(*args, **kwargs, input_regularizer=input_regularizer)

    def set_batchnorm_type(self):
        if not self.rot_eq_batch_norm:
            self.batchnorm_layer_cls = nn.BatchNorm2d
            self.bias_layer_cls = Bias2DLayer
            self.scale_layer_cls = Scale2DLayer
        else:
            self.batchnorm_layer_cls = partial(RotationEquivariantBatchNorm2D, num_rotations=self.num_rotations)
            self.bias_layer_cls = partial(RotationEquivariantBias2DLayer, num_rotations=self.num_rotations)
            self.scale_layer_cls = partial(RotationEquivariantScale2DLayer, num_rotations=self.num_rotations)

    def add_first_layer(self):
        layer = OrderedDict()
        layer["hermite_conv"] = HermiteConv2D(
            input_features=self.input_channels,
            output_features=self.hidden_channels,
            num_rotations=self.num_rotations,
            upsampling=self.upsampling,
            filter_size=self.input_kern,
            stride=self.stride,
            padding=self.input_kern // 2 if self.pad_input else 0,
            first_layer=True,
        )
        self.add_bn_layer(layer)
        self.add_activation(layer)
        self.features.add_module("layer0", nn.Sequential(layer))

    def add_subsequent_layers(self):
        if not isinstance(self.hidden_kern, Iterable):
            self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

        for l in range(1, self.num_layers):
            layer = OrderedDict()

            if self.hidden_padding is None:
                self.hidden_padding = self.hidden_kern[l - 1] // 2

            layer["hermite_conv"] = HermiteConv2D(
                input_features=self.hidden_channels * self.num_rotations,
                output_features=self.hidden_channels,
                num_rotations=self.num_rotations,
                upsampling=self.upsampling,
                filter_size=self.hidden_kern[l - 1],
                stride=self.stride,
                padding=self.hidden_padding,
                first_layer=False,
            )
            self.add_bn_layer(layer)
            self.add_activation(layer)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

    def initialize(self):
        self.apply(self.init_conv_hermite)

    @staticmethod
    def init_conv_hermite(m):
        if isinstance(m, HermiteConv2D):
            nn.init.normal_(m.coeffs.data, std=0.1)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            input_ = feat(input_)
            ret.append(input_)

        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weights_all_rotations, avg=self.use_avg_reg)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.num_layers):
            ret = (
                ret
                + self.features[l]
                .conv.weights_all_rotations.pow(2)
                .sum(3, keepdim=True)
                .sum(2, keepdim=True)
                .sqrt()
                .mean()
            )
        return ret / ((self.num_layers - 1) if self.num_layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels * self.num_rotations


class TransferLearningCore(Core, nn.Module):
    """
    Core based on popular image recognition networks from torchvision such as VGG or AlexNet.
    Can be instantiated as random or pretrained. Core is frozen by default, which can be changed with the fine_tune
    argument.
    """

    def __init__(
        self,
        input_channels,
        tl_model_name,
        layers,
        pretrained=True,
        final_batchnorm=True,
        final_nonlinearity=True,
        momentum=0.1,
        fine_tune=False,
        **kwargs,
    ):
        """
        Args:
            input_channels (int): Number of input channels. 1 if greyscale, 3 if RBG
            tl_model_name (str): Name of the image recognition Transfer Learning model. Possible are all models in
            torchvision, i.e. vgg16, alexnet, ...
            layers (int): Number of layers, i.e. after which layer to cut the original network
            pretrained (boolean): Whether to use a randomly initialized or pretrained network
            final_batchnorm (boolean): Whether to add a batch norm after the final conv layer
            final_nonlinearity (boolean): Whether to add a final nonlinearity (ReLU)
            momentum (float): Momentum term for batch norm. Irrelevant if batch_norm=False
            fine_tune (boolean): Whether to clip gradients before this core or to allow training on the core
            **kwargs:
        """
        if kwargs:
            warnings.warn(
                "Ignoring input {} when creating {}".format(repr(kwargs), self.__class__.__name__),
                UserWarning,
            )
        super().__init__()

        self.input_channels = input_channels
        self.momentum = momentum

        # Download model and cut after specified layer
        TL_model = getattr(torchvision.models, tl_model_name)(pretrained=pretrained)
        TL_model_clipped = nn.Sequential(*list(TL_model.features.children())[:layers])
        if not isinstance(TL_model_clipped[-1], nn.Conv2d):
            warnings.warn(
                "Final layer is of type {}, not nn.Conv2d".format(type(TL_model_clipped[-1])),
                UserWarning,
            )

        # Fix pretrained parameters during training
        if not fine_tune:
            for param in TL_model_clipped.parameters():
                param.requires_grad = False

        # Stack model together
        self.features = nn.Sequential()
        self.features.add_module("TransferLearning", TL_model_clipped)
        if final_batchnorm:
            self.features.add_module("OutBatchNorm", nn.BatchNorm2d(self.outchannels, momentum=self.momentum))
        if final_nonlinearity:
            self.features.add_module("OutNonlin", nn.ReLU(inplace=True))

    def forward(self, input_):
        # If model is designed for RBG input but input is greyscale, repeat the same input 3 times
        if self.input_channels == 1 and self.features.TransferLearning[0].in_channels == 3:
            input_ = input_.repeat(1, 3, 1, 1)
        input_ = self.features(input_)
        return input_

    def regularizer(self):
        return 0

    @property
    def outchannels(self):
        """
        Function which returns the number of channels in the output conv layer. If the output layer is not a conv
        layer, the last conv layer in the network is used.

        Returns: Number of output channels
        """
        found_outchannels = False
        i = 1
        while not found_outchannels:
            if "out_channels" in self.features.TransferLearning[-i].__dict__:
                found_outchannels = True
            else:
                i += 1
        return self.features.TransferLearning[-i].out_channels

    def initialize(self):
        logger.warning(
            "Ignoring initialization since the parameters should be acquired from a pretrained model. If you want random weights, set pretrained = False."
        )


class SE2dCore(Stacked2dCore, nn.Module):
    """
    An extension of the Stacked2dCore class. The convolutional layers can be set to be either depth-separable
    (as used in the popular MobileNets) or based on self-attention (as used in Transformer networks).
    Additionally, a SqueezeAndExcitation layer (also called SE-block) can be added after each layer or the n final
    layers. Finally, it is also possible to make this core fully linear, by disabling all nonlinearities.
    This makes it effectively possible to turn a core+readout CNN into a LNP-model.
    """

    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        bias=False,
        laplace_padding=None,
        input_regularizer="LaplaceL2norm",
        n_se_blocks=0,
        se_reduction=32,
        use_avg_reg=False,
        **kwargs,
    ):
        """
        Args:
            See Stacked2dCore for all input arguments.

            This core provides the functionality to add Squeeze and Excitation Layers, which can be done through
            these additional arguments:

            se_reduction:   Int, Reduction of channels for global pooling of the Squeeze and Excitation Block.
            n_se_blocks:    Int, number of squeeze and excitation blocks. Inserted from the last layer
                              Examples: layers=4, n_se_blocks=2:
                                => layer0 -> layer1 -> layer2 -> SEblock -> layer3 -> SEblock
        """
        self.n_se_blocks = n_se_blocks
        self.se_reduction = se_reduction

        super().__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            input_kern=input_kern,
            hidden_kern=hidden_kern,
            bias=bias,
            laplace_padding=laplace_padding,
            input_regularizer=input_regularizer,
            use_avg_reg=use_avg_reg,
            **kwargs,
        )

    def add_subsequent_layers(self):
        if not isinstance(self.hidden_kern, Iterable):
            self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

        for l in range(1, self.num_layers):
            layer = OrderedDict()
            if self.hidden_padding is None:
                self.hidden_padding = ((self.hidden_kern[l - 1] - 1) * self.hidden_dilation + 1) // 2
            layer[self.conv_layer_name] = self.ConvLayer(
                in_channels=self.hidden_channels if not self.skip > 1 else min(self.skip, l) * self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.hidden_kern[l - 1],
                stride=self.stride,
                padding=self.hidden_padding,
                dilation=self.hidden_dilation,
                bias=self.bias,
            )
            self.add_bn_layer(layer)
            self.add_activation(layer)
            if (self.num_layers - l) <= self.n_se_blocks:
                layer["seg_ex_block"] = SqueezeExcitationBlock(in_ch=self.hidden_channels, reduction=self.se_reduction)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

    def regularizer(self):
        return self.gamma_input * self.laplace()
