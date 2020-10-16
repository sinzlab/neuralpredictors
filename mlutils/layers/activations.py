from torch import nn as nn
from torch.nn import functional as F
import torch
import numpy as np


def elu1(x):
    return F.elu(x, inplace=True) + 1.0


class Elu1(nn.Module):
    """
    Elu activation function shifted by 1 to ensure that the
    output stays positive. That is:
    Elu1(x) = Elu(x) + 1
    """

    def forward(self, x):
        return elu1(x)


def log1exp(x):
    return torch.log(1.0 + torch.exp(x))


class Log1Exp(nn.Module):
    def forward(self, x):
        return log1exp(x)


def adaptive_elu(x, xshift, yshift):
    return F.elu(x - xshift, inplace=True) + yshift


class AdaptiveELU(nn.Module):
    """
    ELU shifted by user specified values. This helps to ensure the output to stay positive.
    """

    def __init__(self, xshift, yshift, **kwargs):
        super(AdaptiveELU, self).__init__(**kwargs)

        self.xshift = xshift
        self.yshift = yshift

    def forward(self, x):
        return adaptive_elu(x, self.xshift, self.yshift)


class PiecewiseLinearExpNonlinearity(nn.Module):
    def __init__(self, number_of_neurons, bias=False, initial_value=0.01, vmin=-3, vmax=6, num_bins=50, smooth_reg_weight=0, smoothnes_reg_order=2):
        super().__init__()
        
        self.bias = bias
        self.initial = initial_value
        self.vmin = vmin
        self.vmax = vmax
        self.neurons = number_of_neurons
        self.smooth_reg_weight = smooth_reg_weight
        self.smoothnes_reg_order = smoothnes_reg_order
        
        self.num_bins = 2 * int(num_bins / 2)
        
        if self.bias:
            self.b = torch.nn.Parameter(torch.empty((number_of_neurons,), dtype=torch.float32).fill_(self.initial))
        self.a = torch.nn.Parameter(torch.empty((self.num_bins, self.neurons), dtype=torch.float32).fill_(0))

        bins = np.linspace(self.vmin, self.vmax, self.num_bins+1, endpoint=True).reshape(1, -1)
        bins_mtx = np.tile(bins, [1, self.neurons, 1])
        bins_mtx = np.transpose(bins_mtx, (0, 2, 1)).astype(np.float32)
        # shape: 1, num_bins, num_neurons
        self.bins = torch.nn.Parameter(torch.from_numpy(bins_mtx), requires_grad=False)
        
        self.zero = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32), requires_grad=False)
        
    def tent(self, x, a, b):
        return torch.min(torch.max(self.zero, x - a), torch.max(self.zero, 2 * b - a - x)) / (b - a)
    
    def linstep(self, x, a, b):
        return torch.min(b - a, torch.max(self.zero, x - a)) / (b - a)
        
    def smoothness_regularizer(self, verbose=False):
        penalty = 0
        kernel = torch.tensor(np.reshape([-1.0, 1.0], (1, 1, 2)), dtype=torch.float32).cuda()
        
        w = torch.reshape(self.a, (-1, 1, self.num_bins)) # shape: neurons, 1, bins
        for k in range(self.smoothnes_reg_order):
            w = F.conv1d(w, kernel)
            penalty += torch.sum(torch.mean(w**2, 1))
        penalty = torch.sum(self.smooth_reg_weight * penalty)
        if verbose:
            print('PieceWiseLinearExpNonLin, Smoothness penalty:', penalty)
        return penalty
    
    def forward(self, x):
        if self.bias:
            # a bias is added
            x = x + self.b
        
        g = torch.nn.functional.elu(x - 1) + 1
        xx = torch.reshape(x, (-1, 1, self.neurons))
        
        # a tent function is applied on the data in multiple bins. 
        tents = self.tent(xx, self.bins[:, :-2, :], self.bins[:, 1:-1, :])
        linstep = self.linstep(xx, self.bins[:, -2:-1, :], self.bins[:, -1:, :])
                               
        t = torch.cat((
                tents, 
                linstep
            ), dim=1)
        # bins shape: 1, num_bins, num_neurons
        # t shape: batch, bins, neurons
        
        h = torch.sum(torch.exp(self.a) * t, dim=1)
        
        return g * h
    
    def visualize(self, vmin=None, vmax=None, iters=1000, show=True, return_fig=False, neurons=range(10)):
        if vmin is None:
            vmin = self.vmin - 1
        if vmax is None:
            vmax = self.vmax + 1
            
        inpts = torch.from_numpy(np.tile(np.linspace(vmin, vmax, iters).astype(np.float32), [self.neurons, 1]).T).cuda()
        outs = self.forward(inpts)
        
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
        ax.plot(inpts.cpu().detach().numpy()[:, neurons], outs.cpu().detach().numpy()[:, neurons])
        ax.set_xlabel('Response before alteration')
        ax.set_ylabel('Response after alteration')
        
        plt.grid(which='both')
        
        if show:
            f.show()
        if return_fig:
            return f
        
class MultiplePiecewiseLinearExpNonlinearity(nn.ModuleDict):
    def __init__(self, n_neurons_dict, bias=False, initial_value=0.01, vmin=-3, vmax=6, num_bins=50, smooth_reg_weight=0, smoothnes_reg_order=2):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            n_neurons = n_neurons_dict[k]
            self.add_module(k, PiecewiseLinearExpNonlinearity(number_of_neurons=n_neurons,
                                            bias=bias,
                                            initial_value=initial_value,
                                            vmin=vmin,
                                            vmax=vmax,
                                            num_bins=num_bins,
                                            smooth_reg_weight=smooth_reg_weight,
                                            smoothnes_reg_order=smoothnes_reg_order)
                            )

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)


    def regularizer(self, data_key):
        return self[data_key].smoothness_regularizer()
