from torch import nn


class DepthSeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
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
