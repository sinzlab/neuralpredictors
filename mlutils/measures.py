from torch import nn
import torch
import numpy as np

class Corr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        target = target.detach()
        delta_out = (output - output.mean(0, keepdim=True))
        delta_target = (target - target.mean(0, keepdim=True))

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
                (var_out + self.eps) * (var_target + self.eps)).sqrt()
        return corrs


class PoissonLoss(nn.Module):
    def __init__(self, bias=1e-12, per_neuron=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron

    def forward(self, output, target):
        target = target.detach()
        loss = (output - target * torch.log(output + self.bias))
        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class PoissonLoss3d(nn.Module):
    def __init__(self, bias=1e-16, per_neuron=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron

    def forward(self, output, target):
        lag = target.size(1) - output.size(1)
        loss =  (output - target[:, lag:, :] * torch.log(output + self.bias))
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
        target = target.detach()
        # use output + 1/2 as shape parameter
        shape = output + 0.5

        # assert np.all(shape.detach().cpu().numpy() > 0), 'Shape parameter is smaller than zero'
        loss = torch.lgamma(shape) - (shape - 1) * torch.log(target + self.bias) + target

        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class AvgCorr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        delta_out = (output - output.mean(0, keepdim=True))
        delta_target = (target - target.mean(0, keepdim=True))

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
                (var_out + self.eps) * (var_target + self.eps)).sqrt()
        return -corrs.mean()


def corr(y1,y2, axis=-1, eps=1e-8, **kwargs):
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
    y1 = (y1 - y1.mean(axis=axis, keepdims=True))/(y1.std(axis=axis, keepdims=True, ddof=1) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True))/(y2.std(axis=axis, keepdims=True, ddof=1) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)

