from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import nn


class Core(ABC):
    """
    Base class for the core models, taking 2d inputs and computing nonlinear features.
    """

    def initialize(self):
        """
        Initialization applied on the core.
        """
        self.apply(self.init_conv)

    @staticmethod
    def init_conv(m):
        """
        Initialize convolution layers with:
            - weights: xavier_normal
            - biases: 0

        Args:
            m (nn.Module): a pytorch nn module.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

    @abstractmethod
    def regularizer(self):
        """
        Regularization applied on the core. Returns a scalar value.
        """

    @abstractmethod
    def forward(self, x):
        """
        Forward function for pytorch nn module.

        Args:
            x (torch.tensor): input of shape (batch, channels, height, width)
        """

    def __repr__(self):
        s = super().__repr__()
        s += f" [{self.__class__.__name__} regularizers: "
        ret = []
        for attr in filter(lambda x: "gamma" in x or "skip" in x, dir(self)):
            ret.append(f"{attr} = {getattr(self, attr)}")
        return s + "|".join(ret) + "]\n"


class ConvCore(Core):
    def __init__(self) -> None:
        """
        Derived classes need to define "batch_norm", "hidden_channels", "independent_bn_bias", "momentum" attributes.
        """
        super().__init__()
        self.set_batchnorm_type()

    @abstractmethod
    def set_batchnorm_type(self):
        """
        Set batchnorm_layer_cls, bias_layer_cls, scale_layer_cls class attributes
        """
        self.batchnorm_layer_cls = None
        self.bias_layer_cls = None
        self.scale_layer_cls = None

    def add_bn_layer(self, layer: OrderedDict, layer_idx: int):
        for attr in ["batch_norm", "hidden_channels", "independent_bn_bias", "momentum"]:
            if not hasattr(self, attr):
                raise NotImplementedError(f"Subclasses must have a `{attr}` attribute.")
        for attr in ["batch_norm", "hidden_channels"]:
            if not isinstance(getattr(self, attr), list):
                raise ValueError(f"`{attr}` must be a list.")

        if self.batch_norm[layer_idx]:
            hidden_channels = self.hidden_channels[layer_idx]

            if self.independent_bn_bias:
                layer["norm"] = self.batchnorm_layer_cls(hidden_channels, momentum=self.momentum)
                return

            bias = self.bias[layer_idx]
            scale = self.batch_norm_scale[layer_idx]

            layer["norm"] = self.batchnorm_layer_cls(hidden_channels, momentum=self.momentum, affine=bias and scale)
            if bias and not scale:
                layer["bias"] = self.bias_layer_cls(hidden_channels)
            elif not bias and scale:
                layer["scale"] = self.scale_layer_cls(hidden_channels)
