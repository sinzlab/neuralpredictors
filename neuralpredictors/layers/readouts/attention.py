import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from .base import Readout


class AttentionReadout(Readout):
    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        init_noise=1e-3,
        attention_kernel=1,
        attention_layers=1,
        mean_activity=None,
        feature_reg_weight=1.0,
        gamma_readout=None,  # depricated, use feature_reg_weight instead
        **kwargs,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(feature_reg_weight, gamma_readout)
        self.mean_activity = mean_activity
        c, w, h = in_shape
        self.features = Parameter(torch.Tensor(self.outdims, c))

        attention = nn.Sequential()
        for i in range(attention_layers - 1):
            attention.add_module(
                f"conv{i}",
                nn.Conv2d(c, c, attention_kernel, padding=attention_kernel > 1),
            )
            attention.add_module(f"norm{i}", nn.BatchNorm2d(c))
            attention.add_module(f"nonlin{i}", nn.ELU())
        else:
            attention.add_module(
                f"conv{attention_layers}",
                nn.Conv2d(c, outdims, attention_kernel, padding=attention_kernel > 1),
            )
        self.attention = attention

        self.init_noise = init_noise
        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)
        self.initialize(mean_activity)

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def initialize_attention(self):
        self.apply(self.init_conv)

    def initialize(self, mean_activity=None):
        if mean_activity is None:
            mean_activity = self.mean_activity
        self.features.data.normal_(0, self.init_noise)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)
        self.initialize_attention()

    def feature_l1(self, reduction="sum", average=None):
        return self.apply_reduction(self.features.abs(), reduction=reduction, average=average)

    def regularizer(self, reduction="sum", average=None):
        return self.feature_l1(reduction=reduction, average=average) * self.feature_reg_weight

    def forward(self, x, shift=None):
        attention = self.attention(x)
        b, c, w, h = attention.shape
        attention = F.softmax(attention.view(b, c, -1), dim=-1).view(b, c, w, h)
        y = torch.einsum("bnwh,bcwh->bcn", attention, x)
        y = torch.einsum("bcn,nc->bn", y, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        return self.__class__.__name__ + " (" + "{} x {} x {}".format(*self.in_shape) + " -> " + str(self.outdims) + ")"
