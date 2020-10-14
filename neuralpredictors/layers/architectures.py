import torch
from torch import nn as nn


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


class SQ_EX_Block(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(SQ_EX_Block, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.view(*(x.shape[:-2]), -1).mean(-1)


class BiasNet(nn.Module):
    """
    Small helper network that adds a learnable bias to an already instantiated base network
    """

    def __init__(self, base_net):
        super(BiasNet, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(2))
        self.base_net = base_net

    def forward(self, x):
        return self.base_net(x) + self.bias
