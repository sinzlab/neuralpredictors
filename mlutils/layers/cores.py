from collections import OrderedDict
from torch import nn
from ..regularizers import LaplaceL2
import torch
import torchvision
import warnings

class Core:
    def initialize(self):
        raise NotImplementedError('Not initializing')

    def __repr__(self):
        s = super().__repr__()
        s += ' [{} regularizers: '.format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: 'gamma' in x or 'skip' in x, dir(self)):
            ret.append('{} = {}'.format(attr, getattr(self, attr)))
        return s + '|'.join(ret) + ']\n'


class Core2d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)
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
    def __init__(self, input_channels, hidden_channels, input_kern, hidden_kern, layers=3,
                 gamma_hidden=0, gamma_input=0., skip=0, final_nonlinearity=True, bias=False,
                 momentum=0.1, pad_input=True, batch_norm=True, hidden_dilation=1):
        super().__init__()

        assert not bias or not batch_norm, "bias and batch_norm should not both be true"
        self._input_weights_regularizer = LaplaceL2()

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.skip = skip

        self.features = nn.Sequential()
        # --- first layer
        layer = OrderedDict()
        layer['conv'] = \
            nn.Conv2d(input_channels, hidden_channels, input_kern,
                      padding=input_kern // 2 if pad_input else 0, bias=bias)
        if batch_norm:
            layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
        if layers > 1 or final_nonlinearity:
            layer['nonlin'] = nn.ELU(inplace=True)
        self.features.add_module('layer0', nn.Sequential(layer))

        # --- other layers
        h_pad = ((hidden_kern - 1) * hidden_dilation + 1) // 2
        for l in range(1, self.layers):
            layer = OrderedDict()
            layer['conv'] = \
                nn.Conv2d(hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                          hidden_channels, hidden_kern,
                          padding=h_pad, bias=bias, dilation=hidden_dilation)
            if batch_norm:
                layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
            if final_nonlinearity or l < self.layers - 1:
                layer['nonlin'] = nn.ELU(inplace=True)
            self.features.add_module('layer{}'.format(l), nn.Sequential(layer))

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l):], dim=1))
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
    def __init__(self, input_channels, tl_model_name, layers, pretrained=True,
                 final_batchnorm=True, final_nonlinearity=True,
                 momentum=0.1, fine_tune=False, **kwargs):
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
            warnings.warn('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__),
                          UserWarning)
        super().__init__()

        self.input_channels = input_channels
        self.momentum = momentum

        # Download model and cut after specified layer
        TL_model = getattr(torchvision.models, tl_model_name)(pretrained=pretrained)
        TL_model_clipped = nn.Sequential(*list(TL_model.features.children())[:layers])
        if not isinstance(TL_model_clipped[-1], nn.Conv2d):
            warnings.warn('Final layer is of type {}, not nn.Conv2d'.format(type(TL_model_clipped[-1])), UserWarning)

        # Fix pretrained parameters during training
        if not fine_tune:
            for param in TL_model_clipped.parameters():
                param.requires_grad = False

        # Stack model together
        self.features = nn.Sequential()
        self.features.add_module('TransferLearning', TL_model_clipped)
        if final_batchnorm:
            self.features.add_module('OutBatchNorm', nn.BatchNorm2d(self.outchannels, momentum=self.momentum))
        if final_nonlinearity:
            self.features.add_module('OutNonlin', nn.ReLU(inplace=True))

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
            if 'out_channels' in self.features.TransferLearning[-i].__dict__:
                found_outchannels = True
            else:
                i += 1
        return self.features.TransferLearning[-i].out_channels
