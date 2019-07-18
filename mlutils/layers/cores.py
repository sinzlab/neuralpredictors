from collections import OrderedDict
from torch import nn
from ..regularizers import LaplaceL2
import torch


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

