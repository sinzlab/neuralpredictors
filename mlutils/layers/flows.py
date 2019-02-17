import math

import numpy as np
import torch
from scipy import linalg
from torch import nn
from torch.nn import functional as F


# TODO: make sure to either compute the inverse log det and add, or the forward logdet and subtract

class Identity(nn.Module):

    def forward(self, input):
        return input


class ResNet(nn.Module):

    def __init__(self, n_in, n_out, layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(n_in if i == 0 else n_out, n_out),
                nn.BatchNorm1d(n_out),
                nn.ELU())
                for i in range(layers)])

    def forward(self, x):
        out = 0
        for layer in self.layers:
            x = layer(x)
            out = out + x
        return out


class InvertibleLinear(nn.Module):
    def __init__(self, pdims, ldims=None, type='full', components=2):
        super().__init__()
        w_shape = [pdims, pdims]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)

        self.nonlinear_bias = ldims is not None
        if ldims is None:
            self.bias = nn.Parameter(torch.zeros(1, pdims))
        else:
            self.bias = ldims

        self._type = type
        self._components = components

        if type == 'full':
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        elif type == 'lowrank':
            self.register_parameter("u", nn.Parameter(torch.zeros(pdims, components)))
            self.register_parameter("v", nn.Parameter(torch.zeros(components, pdims)))
            self.d = nn.Parameter(torch.ones(pdims))
        elif type == 'LU':
            np_p, np_l, np_u = linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.register_buffer('l_mask', torch.Tensor(l_mask))
            self.register_buffer('I', torch.Tensor(eye))

            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))

    def get_weight(self, input, reverse, compute_logdet=False):
        dlogdet = None

        if self._type == 'full':
            if compute_logdet:
                dlogdet = torch.slogdet(self.weight)[1]
            if not reverse:
                weight = self.weight
            else:
                weight = torch.inverse(self.weight.double()).float()

        elif self._type == 'LU':
            l = self.l * self.l_mask + self.I
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            if compute_logdet:
                dlogdet = self.log_s.sum()
            if not reverse:
                weight = self.p @ l @ u
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                weight = u @ l @ self.p.inverse()

        elif self._type == 'lowrank':
            if not reverse:
                weight = self.u @ self.v + torch.diag(self.d)
            else:
                raise NotImplementedError()

            if compute_logdet:
                dlogdet = torch.slogdet(weight)[1]

        return weight, dlogdet


    def forward(self, y, x=None, logdet=None, reverse=False):

        weight, dlogdet = self.get_weight(y, reverse, compute_logdet=logdet is not None)
        if not reverse:
            bias = self.bias if self.nonlinear_bias is None else self.bias(x)
            z = F.linear(y, weight) - bias # <-- the negative bias is important if bias is supposed to be a neuronal prediction
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, x, logdet
        else:
            raise NotImplementedError('Not implemented yet')


class AffineLayer(nn.Module):

    def __init__(self, pdim, ldim):
        super().__init__()
        outdim = pdim // 2 if pdim % 2 == 0 else pdim // 2 + 1
        self.linear = nn.Linear(pdim // 2 + ldim, 2 * outdim)
        self.outdim = outdim
        self.initialize()

    def initialize(self):
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, y, x):
        pred = F.elu(self.linear(torch.cat((y, x), dim=1)))
        return pred[:, :self.outdim], pred[:, self.outdim:]


class AdditiveLayer(nn.Module):

    def __init__(self, pdim, ldim):
        super().__init__()
        outdim = pdim // 2 if pdim % 2 == 0 else pdim // 2 + 1
        self.linear = nn.Linear(pdim // 2 + ldim, outdim)
        self.outdim = outdim
        self.initialize()

    def initialize(self):
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, y, x):
        return F.elu(self.linear(torch.cat((y, x), dim=1)))


class Anscombe(nn.Module):

    def forward(self, y, x=None, logdet=None, reverse=False):

        if not reverse:
            z = 2 * torch.sqrt(y + 3 / 8)

            if logdet is not None:
                dlogdet = (-0.5 * torch.log(y + 3 / 8)).sum(dim=1)
                logdet = logdet + dlogdet
        else:
            raise NotImplementedError('Check above todo')
            # z = (y / 2) ** 2 - 3 / 8
            # if logdet is not None:
            #     dlogdet = (y / 2).log().sum(dim=1)
            #     logdet = logdet - dlogdet

        return z, x, logdet


class Permute(nn.Module):

    def __init__(self, neurons, shuffle=True):
        super().__init__()
        self.neurons = neurons

        if not shuffle:
            indices = np.arange(self.neurons - 1, -1, -1).astype(np.long)
        else:
            indices = np.random.permutation(self.neurons)

        indices_inverse = np.argsort(indices)

        self.register_buffer('indices', torch.LongTensor(indices))
        self.register_buffer('indices_inverse', torch.LongTensor(indices_inverse))

    def forward(self, y, x=None, logdet=None, reverse=False):
        assert len(y.shape) == 2
        if not reverse:
            y = y[:, self.indices]
        else:
            y = y[:, self.indices_inverse]
        return y, x, logdet


def split(tensor, type="split"):
    """
    type = ["middle", "alternate"]
    """

    C = tensor.size(1)
    if type == "middle":
        return tensor[:, :C // 2], tensor[:, C // 2:]
    elif type == "alternate":
        return tensor[:, 0::2], tensor[:, 1::2]


class CouplingLayer(nn.Module):

    def __init__(self, f, split_type='middle', affine=True):
        super().__init__()
        self.f = f
        self.split_type = split_type
        self.affine = affine

    def forward(self, y, x, logdet=None, reverse=False):
        if not reverse:
            y1, y2 = split(y, type=self.split_type)

            if self.affine:
                logs, t = self.f(y1, x)
                s = logs.exp()
                y2 = s * y2 + t

                if logdet is not None:
                    logdet = logdet + logs.sum(dim=1)
            else:
                t = self.f(y1, x)
                y2 = y2 + t

            z = torch.cat((y1, y2), dim=1)
            return z, x, logdet
        else:
            raise NotImplementedError('Inverse not implemented yet.')


# class PoissonPopulationFlow(nn.Module):
#
#     def __init__(self, neurons, bins):
#         super().__init__()
#         self.tuning_curves = nn.Linear(bins, neurons, bias=None)
#         self.anscombe = Anscombe()
#
#     def forward(self, y, x, logdet=None, reverse=False):
#         g = self.tuning_curves(x)
#         A = self.anscombe
#         if not reverse:
#             y, _, logdet = A(y, x, logdet=logdet, reverse=reverse)
#             g, _, _ = A(g)
#             z = y - g - 1 / (4 * g.sqrt())
#         else:
#             raise NotImplementedError('Sampling not implemented yet')
#         return z, g, logdet


class ConditionalFlow(nn.ModuleList):
    LOG2 = math.log(2)

    def __init__(self, modules, output_dim):
        super().__init__(modules)
        self.output_dim = output_dim

    def forward(self, y, x=None, logdet=None, reverse=False):
        modules = self if not reverse else reversed(self)
        for module in modules:
            y, x, logdet = module(y, x=x, logdet=logdet, reverse=reverse)
        return y, x, logdet

    def cross_entropy(self, y, x, target_density, average=True):
        z, _, ld = self(y, x, logdet=0)
        if average:
            return -(target_density.log_prob(z).mean() + ld.mean()) / self.output_dim / self.LOG2
        else:
            return -(target_density.log_prob(z) + ld) / self.output_dim / self.LOG2
