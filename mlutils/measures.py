from torch import nn
import torch
import numpy as np
import warnings


class Corr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        target = target.detach()
        delta_out = output - output.mean(0, keepdim=True)
        delta_target = target - target.mean(0, keepdim=True)

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()
        return corrs


class PoissonLoss(nn.Module):
    def __init__(self, bias=1e-12, per_neuron=False, avg=True):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron
        self.avg = avg
        if self.avg:
            warnings.warn("Poissonloss is averaged per batch. It's recommended so use sum instead")

    def forward(self, output, target):
        target = target.detach()
        loss = output - target * torch.log(output + self.bias)
        if not self.per_neuron:
            return loss.mean() if self.avg else loss.sum()
        else:
            loss = loss.view(-1, loss.shape[-1])
            return loss.mean(dim=0) if self.avg else loss.sum(dim=0)


class PoissonLoss3d(nn.Module):
    def __init__(self, bias=1e-16, per_neuron=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron

    def forward(self, output, target):
        lag = target.size(1) - output.size(1)
        loss = output - target[:, lag:, :] * torch.log(output + self.bias)
        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class GammaLoss(nn.Module):
    def __init__(self, bias=1e-12, per_neuron=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron

    def forward(self, output, target):
        target = (target + self.bias).detach()
        # use output + 1/2 as shape parameter
        shape = output + 0.5

        # assert np.all(shape.detach().cpu().numpy() > 0), 'Shape parameter is smaller than zero'
        loss = torch.lgamma(shape) - (shape - 1) * torch.log(target) + target

        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class ExponentialLoss(nn.Module):
    def __init__(self, bias=1e-12, target_bias=1e-6, per_neuron=False):
        super().__init__()
        self.bias = bias
        self.target_bias = target_bias
        self.per_neuron = per_neuron

    def forward(self, output, target):
        output = output + self.bias

        target = (target + self.target_bias).detach()

        loss = target / output + torch.log(output)

        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class AnscombeMSE(nn.Module):
    def __init__(self, per_neuron=False):
        super().__init__()
        self.per_neuron = per_neuron

    @staticmethod
    def A(x):
        return 2 * torch.sqrt(x + 3 / 8)

    def forward(self, output, target):
        target = self.A(target).detach()
        output = self.A(output) - 1 / (4 * output.sqrt())

        loss = (target - output).pow(2)

        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class AvgCorr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        delta_out = output - output.mean(0, keepdim=True)
        delta_target = target - target.mean(0, keepdim=True)

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()
        return -corrs.mean()


def corr(y1, y2, axis=-1, eps=1e-8, **kwargs):
    """
    Compute the correlation between two matrices along certain dimensions.

    Args:
        y1:      first matrix
        y2:      second matrix
        axis:    dimension along which the correlation is computed.
        eps:     offset to the standard deviation to make sure the correlation is well defined (default 1e-8)
        **kwargs passed to final mean of standardized y1 * y2

    Returns: correlation vector

    """
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=0) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=0) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)
