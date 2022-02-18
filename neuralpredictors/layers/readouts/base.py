import warnings
from typing import Any, Literal, Mapping, Optional

import torch
from torch import nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter

Reduction = Literal["sum", "mean", None]


class ConfigurationError(Exception):
    pass


# ------------------ Base Classes -------------------------


class Readout(Module):
    """
    Base readout class for all individual readouts.
    The MultiReadout will expect its readouts to inherit from this base class.
    """

    features: Parameter
    bias: Parameter

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def regularizer(self, reduction: Reduction = "sum", average: Optional[bool] = None) -> torch.Tensor:
        raise NotImplementedError("regularizer is not implemented for ", self.__class__.__name__)

    def apply_reduction(
        self, x: torch.Tensor, reduction: Reduction = "mean", average: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Applies a reduction on the output of the regularizer.
        Args:
            x: output of the regularizer
            reduction: method of reduction for the regularizer. Currently possible are ['mean', 'sum', None].
            average: Deprecated. Whether to average the output of the regularizer.
                            If not None, it is transformed into the corresponding value of 'reduction' (see method 'resolve_reduction_method').

        Returns: reduced value of the regularizer
        """
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)

        if reduction == "mean":
            return x.mean()
        elif reduction == "sum":
            return x.sum()
        elif reduction is None:
            return x
        else:
            raise ValueError(
                f"Reduction method '{reduction}' is not recognized. Valid values are ['mean', 'sum', None]"
            )

    def resolve_reduction_method(self, reduction: Reduction = "mean", average: Optional[bool] = None) -> Reduction:
        """
        Helper method which transforms the old and deprecated argument 'average' in the regularizer into
        the new argument 'reduction' (if average is not None). This is done in order to agree with the terminology in pytorch).
        """
        if average is not None:
            warnings.warn("Use of 'average' is deprecated. Please consider using `reduction` instead")
            reduction = "mean" if average else "sum"
        return reduction

    def resolve_deprecated_gamma_readout(self, feature_reg_weight: float, gamma_readout: Optional[float]) -> float:
        if gamma_readout is not None:
            warnings.warn(
                "Use of 'gamma_readout' is deprecated. Please consider using the readout's feature-regularization parameter instead"
            )
            feature_reg_weight = gamma_readout
        return feature_reg_weight

    def initialize_bias(self, mean_activity: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the biases in readout.
        Args:
            mean_activity: Tensor containing the mean activity of neurons.

        Returns:

        """
        if mean_activity is None:
            warnings.warn("Readout is NOT initialized with mean activity but with 0!")
            self.bias.data.fill_(0)
        else:
            self.bias.data = mean_activity

    def __repr__(self) -> str:
        return super().__repr__() + " [{}]\n".format(self.__class__.__name__)  # type: ignore[no-untyped-call,no-any-return]


class ClonedReadout(Module):
    """
    This readout clones another readout while applying a linear transformation on the output. Used for MultiDatasets
    with matched neurons where the x-y positions in the grid stay the same but the predicted responses are rescaled due
    to varying experimental conditions.
    """

    def __init__(self, original_readout: Readout, **kwargs: Any) -> None:
        super().__init__()  # type: ignore[no-untyped-call]

        self._source = original_readout
        self.alpha = Parameter(torch.ones(self._source.features.shape[-1]))  # type: ignore[attr-defined]
        self.beta = Parameter(torch.zeros(self._source.features.shape[-1]))  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor, **kwarg: Any) -> torch.Tensor:
        x = self._source(x) * self.alpha + self.beta
        return x

    def feature_l1(self, average: bool = True) -> torch.Tensor:
        """Regularization is only applied on the scaled feature weights, not on the bias."""
        if average:
            return (self._source.features * self.alpha).abs().mean()
        else:
            return (self._source.features * self.alpha).abs().sum()

    def initialize(self, **kwargs: Any) -> None:
        self.alpha.data.fill_(1.0)
        self.beta.data.fill_(0.0)
