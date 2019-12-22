import warnings
from collections import OrderedDict, Iterable

from torch import nn
import torch
import torchvision

from .. import regularizers


class Core:
    def initialize(self):
        raise NotImplementedError("Not initializing")

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: "gamma" in x or "skip" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


class Core2d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)
        self.put_to_cuda(cuda=cuda)

    def put_to_cuda(self, cuda):
        if cuda:
            self = self.cuda()

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.fill_(0)


# ---------------------- Conv2d Cores -----------------------------


class Stacked2dCore(Core2d, nn.Module):
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
        final_nonlinearity=True,
        bias=False,
        momentum=0.1,
        pad_input=True,
        hidden_padding=None,
        batch_norm=True,
        hidden_dilation=1,
        laplace_padding=0,
        input_regularizer="LaplaceL2",
        stack=None,
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            gamma_hidden:   regularizer factor for group sparsity
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see mlutils.regularizers)
            skip:           Adds a skip connection
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            bias:           Adds a bias layer. Note: bias and batch_norm can not both be true
            momentum:       BN momentum
            pad_input:      Boolean, if True, applies zero padding to all convolutions
            hidden_padding: int or list of int. Padding for hidden layers. Note that this will apply to all the layers 
                            except the first (input) layer.
            batch_norm:     Boolean, if True appends a BN layer after each convolutional layer
            hidden_dilation:    If set to > 1, will apply dilated convs for all hidden layers
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            normalize_laplace_regularizer: Boolean, if set to True, will use the LaplaceL2norm function from
                mlutils.regularizers, which returns the regularizer as |laplace(filters)| / |filters|
            input_regularizer:  String that must match one of the regularizers in ..regularizers
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            stack = -1 will only select the last layer as the readout layer
                            stack = 0  will only readout from the first layer
        """

        super().__init__()

        assert not bias or not batch_norm, "bias and batch_norm should not both be true"

        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kern)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.skip = skip
        self.features = nn.Sequential()
        if stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = [range(self.layers)[stack]] if isinstance(stack, int) else stack

        # --- first layer
        layer = OrderedDict()
        layer["conv"] = nn.Conv2d(
            input_channels, hidden_channels, input_kern, padding=input_kern // 2 if pad_input else 0, bias=bias
        )
        if batch_norm:
            layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
        if layers > 1 or final_nonlinearity:
            layer["nonlin"] = nn.ELU(inplace=True)
        self.features.add_module("layer0", nn.Sequential(layer))

        # --- other layers
        if hidden_padding is None:
            hidden_padding = ((hidden_kern - 1) * hidden_dilation + 1) // 2

        if not isinstance(hidden_padding, Iterable):
            hidden_padding = [hidden_padding] * (self.layers - 1)

        if not isinstance(hidden_kern, Iterable):
            hidden_kern = [hidden_kern] * (self.layers - 1)

        for l in range(1, self.layers):
            layer = OrderedDict()
            layer["conv"] = nn.Conv2d(
                hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                hidden_channels,
                hidden_kern[l-1],
                padding=hidden_padding[l-1],
                bias=bias,
                dilation=hidden_dilation,
            )
            if batch_norm:
                layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
            if final_nonlinearity or l < self.layers - 1:
                layer["nonlin"] = nn.ELU(inplace=True)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l):], dim=1))
            if l in self.stack:
                ret.append(input_)
        return torch.cat(ret, dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels


class TransferLearningCore(Core2d, nn.Module):
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
        **kwargs
    ):
        """
        Core from popular image recognition networks such as VGG or AlexNet. Can be already pretrained on ImageNet.

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
                "Ignoring input {} when creating {}".format(repr(kwargs), self.__class__.__name__), UserWarning
            )
        super().__init__()

        self.input_channels = input_channels
        self.momentum = momentum

        # Download model and cut after specified layer
        TL_model = getattr(torchvision.models, tl_model_name)(pretrained=pretrained)
        TL_model_clipped = nn.Sequential(*list(TL_model.features.children())[:layers])
        if not isinstance(TL_model_clipped[-1], nn.Conv2d):
            warnings.warn("Final layer is of type {}, not nn.Conv2d".format(type(TL_model_clipped[-1])), UserWarning)

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

    def initialize(self, cuda=False):
        # Overwrite parent class's initialize function because initialization is done by the 'pretrained' parameter
        self.put_to_cuda(cuda=cuda)


class DepthSeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.add_module("in_depth_conv", nn.Conv2d(in_channels, out_channels, 1, bias=bias))
        self.add_module(
            "spatial_conv",
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                groups=out_channels,
            ),
        )
        self.add_module("out_depth_conv", nn.Conv2d(out_channels, out_channels, 1, bias=bias))
