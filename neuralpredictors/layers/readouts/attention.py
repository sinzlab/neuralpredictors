from typing import Any, Mapping, Optional, Tuple

import torch
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules import ELU, BatchNorm2d, Conv2d, Module, Sequential
from torch.nn.parameter import Parameter

from .base import Readout, Reduction


class AttentionReadout(Readout):
    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        outdims: int,
        bias: bool,
        init_noise: float = 1e-3,
        attention_kernel: int = 1,
        attention_layers: int = 1,
        mean_activity: Optional[torch.Tensor] = None,
        feature_reg_weight: float = 1.0,
        gamma_readout: Optional[float] = None,  # deprecated, use feature_reg_weight instead
        **kwargs: Any,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.in_shape = in_shape
        self.outdims = outdims
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(feature_reg_weight, gamma_readout)
        self.mean_activity = mean_activity
        c, w, h = in_shape
        self.features = Parameter(torch.Tensor(self.outdims, c))

        attention = Sequential()
        for i in range(attention_layers - 1):
            attention.add_module(
                f"conv{i}",
                Conv2d(c, c, attention_kernel, padding=attention_kernel > 1),
            )
            attention.add_module(f"norm{i}", BatchNorm2d(c))  # type: ignore[no-untyped-call]
            attention.add_module(f"nonlin{i}", ELU())
        else:
            attention.add_module(
                f"conv{attention_layers}",
                Conv2d(c, outdims, attention_kernel, padding=attention_kernel > 1),
            )
        self.attention = attention

        self.init_noise = init_noise
        if bias:
            bias_param = Parameter(torch.Tensor(self.outdims))
            self.register_parameter("bias", bias_param)
        else:
            self.register_parameter("bias", None)
        self.initialize(mean_activity)

    @staticmethod
    def init_conv(m: Module) -> None:
        if isinstance(m, Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def initialize_attention(self) -> None:
        self.apply(self.init_conv)

    def initialize(self, mean_activity: Optional[torch.Tensor] = None) -> None:  # type: ignore[override]
        if mean_activity is None:
            mean_activity = self.mean_activity
        self.features.data.normal_(0, self.init_noise)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)
        self.initialize_attention()

    def feature_l1(self, reduction: Reduction = "sum", average: Optional[bool] = None) -> torch.Tensor:
        return self.apply_reduction(self.features.abs(), reduction=reduction, average=average)

    def regularizer(self, reduction: Reduction = "sum", average: Optional[bool] = None) -> torch.Tensor:
        return self.feature_l1(reduction=reduction, average=average) * self.feature_reg_weight

    def forward(self, x: torch.Tensor, shift: Optional[Any] = None) -> torch.Tensor:
        attention = self.attention(x)
        b, c, w, h = attention.shape
        attention = F.softmax(attention.view(b, c, -1), dim=-1).view(b, c, w, h)
        y: torch.Tensor = torch.einsum("bnwh,bcwh->bcn", attention, x)  # type: ignore[attr-defined]
        y = torch.einsum("bcn,nc->bn", y, self.features)  # type: ignore[attr-defined]
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self) -> str:
        return self.__class__.__name__ + " (" + "{} x {} x {}".format(*self.in_shape) + " -> " + str(self.outdims) + ")"
