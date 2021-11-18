import warnings
import torch
from torch import nn as nn
from torch.nn import Parameter


class ConfigurationError(Exception):
    pass


# ------------------ Base Classes -------------------------


class Readout(nn.Module):
    """
    Base readout class for all individual readouts.
    The MultiReadout will expect its readouts to inherit from this base class.
    """

    def initialize(self, *args, **kwargs):
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def regularizer(self, reduction="sum", average=None):
        raise NotImplementedError("regularizer is not implemented for ", self.__class__.__name__)

    def apply_reduction(self, x, reduction="mean", average=None):
        """
        Applies a reduction on the output of the regularizer.
        Args:
            x (torch.tensor): output of the regularizer
            reduction(str/None): method of reduction for the regularizer. Currently possible are ['mean', 'sum', None].
            average (bool): Depricated. Whether to average the output of the regularizer.
                            If not None, it is transformed into the corresponding value of 'reduction' (see method 'resolve_reduction_method').

        Returns (torch.tensor): reduced value of the regularizer
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

    def resolve_reduction_method(self, reduction="mean", average=None):
        """
        Helper method which transforms the old and depricated argument 'average' in the regularizer into
        the new argument 'reduction' (if average is not None). This is done in order to agree with the terminology in pytorch).
        """
        if average is not None:
            warnings.warn("Use of 'average' is deprecated. Please consider using `reduction` instead")
            reduction = "mean" if average else "sum"
        return reduction

    def resolve_deprecated_gamma_readout(self, feature_reg_weight, gamma_readout):
        if gamma_readout is not None:
            warnings.warn(
                "Use of 'gamma_readout' is deprecated. Please consider using the readout's feature-regularization parameter instead"
            )
            feature_reg_weight = gamma_readout
        return feature_reg_weight

    def initialize_bias(self, mean_activity=None):
        """
        Initialize the biases in readout.
        Args:
            mean_activity (dict): Dictionary containing the mean activity of neurons for a specific dataset.
            Should be of form {'data_key': mean_activity}

        Returns:

        """
        if mean_activity is None:
            warnings.warn("Readout is NOT initialized with mean activity but with 0!")
            self.bias.data.fill_(0)
        else:
            self.bias.data = mean_activity

    def __repr__(self):
        return super().__repr__() + " [{}]\n".format(self.__class__.__name__)


class ClonedReadout(nn.Module):
    """
    This readout clones another readout while applying a linear transformation on the output. Used for MultiDatasets
    with matched neurons where the x-y positions in the grid stay the same but the predicted responses are rescaled due
    to varying experimental conditions.
    """

    def __init__(self, original_readout, **kwargs):
        super().__init__()

        self._source = original_readout
        self.alpha = Parameter(torch.ones(self._source.features.shape[-1]))
        self.beta = Parameter(torch.zeros(self._source.features.shape[-1]))

    def forward(self, x):
        x = self._source(x) * self.alpha + self.beta
        return x

    def feature_l1(self, average=True):
        """Regularization is only applied on the scaled feature weights, not on the bias."""
        if average:
            return (self._source.features * self.alpha).abs().mean()
        else:
            return (self._source.features * self.alpha).abs().sum()

    def initialize(self, **kwargs):
        self.alpha.data.fill_(1.0)
        self.beta.data.fill_(0.0)
