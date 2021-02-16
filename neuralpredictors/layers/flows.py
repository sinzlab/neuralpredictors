import math

import numpy as np
import torch
from scipy import linalg
from torch import nn
from torch.nn import functional as F

# TODO: make sure to either compute the inverse log det and add, or the forward logdet and subtract
# TODO: how should x be handled when reverse = True
from ..constraints import positive, at_least


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


class ResNet(nn.Module):
    def __init__(self, n_in, n_out, layers=3, final_linear=False, init_noise=1e-6, n_hidden=None):
        """
        Fully connectected ResNet layer. Each block consists of ELU(BatchNorm(Linear(x)))

        Args:
            n_in:   input dimensions
            n_out:  ouput dimensions
            layers: number of layers

        """
        super().__init__()
        self.final_linear = final_linear
        if not final_linear:
            layers = [nn.Sequential(nn.Linear(n_in if i == 0 else n_out, n_out), nn.ELU()) for i in range(layers)]
        else:
            assert n_hidden is not None, "n_hidden must not bet none when final_linear=True"
            layers = [nn.Sequential(nn.Linear(n_in if i == 0 else n_hidden, n_hidden), nn.ELU()) for i in range(layers)]
            layers.append(nn.Sequential(nn.Linear(n_hidden if len(layers) > 0 else n_hidden, n_out)))

        self.layers = nn.ModuleList(layers)

        for layer in self.layers:
            layer[0].weight.data.normal_(0, init_noise)

    def forward(self, x):
        out = 0

        for layer in self.layers if not self.final_linear else self.layers[:-1]:
            x = layer(x)
            out = out + x
        if self.final_linear:
            out = self.layers[-1](out)
        return out


class InvertibleLinear(nn.Module):
    def __init__(self, pdims, bias=None, type="full", components=2):
        """
        Invertible Linear Layer

        z = Wy + bias(x)

        Args:
            pdims:  invertible (probability) dimensions
            bias:   if None -> no bias, if True -> learned bias, else callable providing bias
            type:   "LU", "full", or "lowrank":
                       - "LU" see Glow paper parametrization,
                       - "full" unconstrained
                       - "lowrank" U @ V + diag(D) (U and V.T have dimension pdims x components)
            components: number of components for "lowrank"
        """
        super().__init__()
        w_shape = [pdims, pdims]
        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)

        self.nonlinear_bias = bias is not None
        if bias is True:
            self.bias = nn.Parameter(torch.zeros(1, pdims))
        elif bias is not None:
            self.bias = bias

        self._type = type
        self._components = components

        if type == "full":
            self.weight = nn.Parameter(torch.Tensor(w_init))
        elif type == "lowrank":
            self.u = nn.Parameter(torch.zeros(pdims, components))  # .normal_() * 1e-2))
            self.v = nn.Parameter(torch.zeros(components, pdims))  # .normal_() * 1e-2))
            self.d = nn.Parameter(torch.ones(pdims))
        elif type == "LU":
            np_p, np_l, np_u = linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer("p", torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer("sign_s", torch.Tensor(np_sign_s.astype(np.float32)))
            self.register_buffer("l_mask", torch.Tensor(l_mask))
            self.register_buffer("I", torch.Tensor(eye))

            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))

    def get_weight(self, reverse, compute_logdet=False):
        dlogdet = None

        if self._type == "full":
            if compute_logdet:
                dlogdet = torch.slogdet(self.weight)[1]

            if not reverse:
                weight = self.weight
            else:
                weight = torch.inverse(self.weight.double()).float()

        elif self._type == "LU":
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

        elif self._type == "lowrank":
            weight = self.u @ self.v + torch.diag(self.d)

            if reverse:
                weight = torch.inverse(weight.double()).float()

            if compute_logdet:
                dlogdet = torch.slogdet(weight)[1]

        if reverse:
            dlogdet = -dlogdet

        return weight, dlogdet

    def forward(self, y, x=None, logdet=None, reverse=False):

        weight, dlogdet = self.get_weight(reverse, compute_logdet=logdet is not None)
        if self.bias:
            bias = self.bias if self.nonlinear_bias is None else self.bias(x)
        else:
            bias = 0

        if not reverse:
            # the minus on the bias is important if the bias is a nonelinear network with positive output
            z = F.linear(y, weight) - bias
        else:
            z = F.linear(y + bias, weight)

        if logdet is not None:
            logdet = logdet + dlogdet

        return z, x, logdet

    def __repr__(self):
        if self._type == "full":
            tstring = "W"
        elif self._type == "LU":
            tstring = "PL(U+ diag(s))"
        elif self._type == "lowrank":
            tstring = "UV + D [components={}]".format(self._components)
        bias = "b" if self.nonlinear_bias is None else "{}(x)".format(self.bias.__class__.__name__)
        return "Linear(W={}, bias={})".format(tstring, bias)


class Anscombe(nn.Module):
    def __init__(self, transform_x=False):
        """
        Variance stabilizing transform for Poisson variables.

        A(x): x |-> 2 * sqrt(x + 3 / 8)

        For Poisson variables with rate r, A(x) is approximately Gaussian with mean A(r) - 1/(4 sqrt(r))
        and standard deviation one.

        Args:
            transform_x: if True, x is transformed with A(x) - 1/(4 sqrt(x))
        """
        super().__init__()
        self.transform_x = transform_x

    def forward(self, y, x=None, logdet=None, reverse=False):
        A = lambda g: 2 * torch.sqrt(g + 3 / 8)
        if not reverse:
            z = A(y)
            if self.transform_x:
                g = A(x)
                x = g - 1 / (4 * g.sqrt())
            if logdet is not None:
                dlogdet = (-0.5 * torch.log(y + 3 / 8)).sum(dim=1)
        else:
            z = 1 / 8 * (2 * y ** 2 - 3)
            if logdet is not None:
                dlogdet = (y / 2).log().sum(dim=1)

        return z, x, logdet if logdet is None else logdet + dlogdet

    def __repr__(self):
        return "Anscombe(transform_x={})".format(self.transform_x)


class Difference(nn.Module):
    def forward(self, y, x, logdet=None, reverse=False):

        if not reverse:
            z = y - x
        else:
            z = y + x

        return z, x, logdet

    def __repr__(self):
        return "Difference(y-x)"


class MixtureOfSigmoids(nn.Module):
    def __init__(self, pdims, components, a_init=0.01, b_init=1.0, scale=1, offset=0):
        super().__init__()
        self.a = nn.Parameter(torch.Tensor(1, pdims, components).uniform_(a_init, 2 * a_init))
        self.b = nn.Parameter(torch.Tensor(1, pdims, components).normal_(0, b_init))
        self.logits = nn.Parameter(torch.Tensor(1, pdims, components).normal_())
        self.pdims = pdims
        self.components = components
        self.scale = scale
        self.offset = offset

    def forward(self, y, x, logdet=None, reverse=False):
        positive(self.a)

        if not reverse:
            f = torch.sigmoid(y[..., None] * self.a + self.b)
            q = F.softmax(self.logits, dim=-1)

            if logdet is not None:
                diag_jacobian = self.scale * (f * (1 - f) * self.a * q).sum(-1)
                logdet = logdet + torch.log(diag_jacobian).sum(1)
            z = (f * q).sum(-1) * self.scale + self.offset
        else:
            raise NotImplementedError()

        return z, x, logdet

    def __repr__(self):
        return "Mixture of Sigmoids(components={}, dimensions={})".format(self.components, self.pdims)


class AffineLog(nn.Module):
    def __init__(self, pdims, lower_bound=1e-4, transform_x=False):
        super().__init__()
        assert lower_bound > 0, "lower bound must be greater than 0"
        self.a = nn.Parameter(torch.ones(1, pdims))
        self.b = nn.Parameter(torch.ones(1, pdims))
        self.lower_bound = lower_bound
        self.pdims = pdims
        self.transform_x = transform_x

    def forward(self, y, x, logdet=None, reverse=False):
        at_least(self.a, self.lower_bound)
        at_least(self.b, self.lower_bound)

        if not reverse:
            act = self.b + self.a * y
            z = torch.log(act)

            if self.transform_x:
                x = torch.log(self.b + self.a * x)

            if logdet is not None:
                log_diag_jacobian = torch.log(self.a) - torch.log(act)
                logdet = logdet + log_diag_jacobian.sum(1)
        else:
            raise NotImplementedError()

        return z, x, logdet

    def __repr__(self):
        return "AffineLog(lower_bound={}, dimensions={})".format(self.lower_bound, self.pdims)


class Logit(nn.Module):
    def forward(self, y, x=None, logdet=None, reverse=False):
        z = torch.log(y / (1 - y))
        if logdet is not None:
            dlogdet = -torch.log(y - y.pow(2))
            logdet = logdet + dlogdet.sum(1)

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

        self.register_buffer("indices", torch.LongTensor(indices))
        self.register_buffer("indices_inverse", torch.LongTensor(indices_inverse))

    def forward(self, y, x=None, logdet=None, reverse=False):
        assert len(y.shape) == 2
        if not reverse:
            y = y[:, self.indices]
        else:
            y = y[:, self.indices_inverse]
        return y, x, logdet


def split(tensor, type="split"):
    """
    Splits a tensor into two along the columns.

    Args:
        tensor: m x dim tensor to be split
        type: Either "middle" (separate at dim//2) or "alternate" (separate every other).
    """

    C = tensor.size(1)
    if type == "middle":
        return tensor[:, : C // 2], tensor[:, C // 2 :]
    elif type == "alternate":
        return tensor[:, 0::2], tensor[:, 1::2]


class Preprocessor(nn.Module):
    def __init__(self, preprocessor):
        """
        Preprocessor module for x (the variables conditioned on).
        Only transforms x and leaves y untouched.

        Args:
            preprocessor: arbitrary nonlinear network preprocessing x
        """
        super().__init__()
        self.preprocessor = preprocessor

    def forward(self, y, x, logdet=None, reverse=False):
        x = self.preprocessor(x)
        return y, x, logdet

    def __repr__(self):
        return "Preprocessor({})".format(self.preprocessor.__name__)


class AffineLayer1D(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, y, x, logdet=None, reverse=False):
        if not reverse:

            tmp = self.f(x)
            logs, t = tmp[:, 0], tmp[:, 1]
            s = logs.exp()
            z = s[:, None] * y + t[:, None]
            # print('scale', logs.detach().numpy())
            if logdet is not None:
                logdet = logdet + logs

            return z, x, logdet
        else:
            raise NotImplementedError("Inverse not implemented yet.")

    def __repr__(self):
        bias = "{}(x)".format(self.f.__class__.__name__)
        return "Linear(coefficients from {})".format(bias)


class Sigmoid1D(nn.Module):
    def __init__(self, f, scale=1, offset=0):
        super().__init__()
        self.f = f
        self.scale = scale
        self.offset = offset

    def forward(self, y, x, logdet=None, reverse=False):
        if not reverse:

            tmp = self.f(x)
            # print('tmp', tmp)
            logs, t = tmp[:, 0], tmp[:, 1]
            s = logs.exp()
            z = torch.sigmoid(s[:, None] * y + t[:, None])

            if logdet is not None:
                dlogdet = math.log(self.scale) + z.squeeze().log() + (1 - z).squeeze().log() + logs
                logdet = logdet + dlogdet

            return self.scale * z + self.offset, x, logdet
        else:
            raise NotImplementedError("Inverse not implemented yet.")

    def __repr__(self):
        bias = "{}(x)".format(self.f.__class__.__name__)
        return "{} * Sigmoid(coefficients from {}) + {}".format(self.scale, bias, self.offset)


class ELU1D(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, y, x, logdet=None, reverse=False):
        if not reverse:
            tmp = self.f(x)
            assert tmp.shape[1] == 2, "f needs to output exactly two dimensions"
            s, t = tmp[:, 0], tmp[:, 1]
            z = F.elu(s[:, None] * y + t[:, None])

            if logdet is not None:
                deludz = (z.exp() - 1) * (z < 0).type(torch.FloatTensor) + 1

                dlogdet = deludz.squeeze().log() + s.abs().log()
                logdet = logdet + dlogdet

            return z, x, logdet
        else:
            raise NotImplementedError("Inverse not implemented yet.")

    def __repr__(self):
        bias = "{}(x)".format(self.f.__class__.__name__)
        return "ELU(coefficients from {})".format(bias)


def bucketize(tensor, bucket_boundaries):
    result = torch.zeros_like(tensor, dtype=torch.long) - 1
    for b in range(bucket_boundaries.shape[-1]):
        result += (tensor >= bucket_boundaries[..., b : b + 1]).long()
    return result.squeeze()


class Interpolate1D(nn.Module):
    def __init__(self, f, resolution, y_min=0, y_max=1, scale=1, offset=0):
        super().__init__()
        self.f = f
        self.scale = scale
        self.offset = offset
        self.resolution = resolution
        self.y_min, self.y_max = (y_min, y_max)
        self.register_buffer("base_points", torch.linspace(self.y_min, self.y_max, resolution))

    def forward(self, y, x, logdet=None, reverse=False):

        basefeatures = F.softmax(self.f(x), dim=1).cumsum(dim=1)

        i = torch.arange(0, basefeatures.shape[0])

        if not reverse:
            start = bucketize(y, self.base_points)
            # start[start < 0] = 0
            # start[start > self.resolution - 2] = self.resolution - 2

            x0 = self.base_points[start][:, None]
            x1 = self.base_points[start + 1][:, None]

            f0 = basefeatures[i, start][:, None]
            f1 = basefeatures[i, start + 1][:, None]

            slope = (f1 - f0) / (x1 - x0)
            z = f0 + slope * (y - x0)
            z = self.scale * z + self.offset

            if logdet is not None:
                dlogdet = math.log(self.scale) + slope.squeeze().abs().log()
                logdet = logdet + dlogdet

            return z, x, logdet
        else:
            y = (y - self.offset) / self.scale
            start = bucketize(y, basefeatures)
            # start[start < 0] = 0
            # start[start > self.resolution - 2] = self.resolution - 2

            f0 = self.base_points[start][:, None]
            f1 = self.base_points[start + 1][:, None]

            x0 = basefeatures[i, start][:, None]
            x1 = basefeatures[i, start + 1][:, None]
            slope = (f1 - f0) / (x1 - x0)
            z = f0 + slope * (y - x0)

            if logdet is not None:
                dlogdet = math.log(self.scale) - slope.squeeze().abs().log()
                logdet = logdet + dlogdet

            return z, x, logdet

    def __repr__(self):
        bias = "{}(x)".format(self.f.__class__.__name__)
        return "{} * Interpolate(base features from {}) + {}".format(self.scale, bias, self.offset)


# class AffineLayer(nn.Module):
#
#     def __init__(self, pdim, ldim):
#         super().__init__()
#         outdim = pdim // 2 if pdim % 2 == 0 else pdim // 2 + 1
#         self.linear = nn.Linear(pdim // 2 + ldim, 2 * outdim)
#         self.outdim = outdim
#         self.initialize()
#
#     def initialize(self):
#         self.linear.weight.data.zero_()
#         self.linear.bias.data.zero_()
#
#     def forward(self, y, x):
#         pred = F.elu(self.linear(torch.cat((y, x), dim=1)))
#         return pred[:, :self.outdim], pred[:, self.outdim:]
#
#
# class AdditiveLayer(nn.Module):
#
#     def __init__(self, pdim, ldim):
#         super().__init__()
#         outdim = pdim // 2 if pdim % 2 == 0 else pdim // 2 + 1
#         self.linear = nn.Linear(pdim // 2 + ldim, outdim)
#         self.outdim = outdim
#         self.initialize()
#
#     def initialize(self):
#         self.linear.weight.data.zero_()
#         self.linear.bias.data.zero_()
#
#     def forward(self, y, x):
#         return F.elu(self.linear(torch.cat((y, x), dim=1)))
#

#
# class CouplingLayer(nn.Module):
#
#     def __init__(self, f, split_type='middle', affine=True):
#         super().__init__()
#         self.f = f
#         self.split_type = split_type
#         self.affine = affine
#
#     def forward(self, y, x, logdet=None, reverse=False):
#         if not reverse:
#             y1, y2 = split(y, type=self.split_type)
#
#             if self.affine:
#                 logs, t = self.f(y1, x)
#                 s = logs.exp()
#                 y2 = s * y2 + t
#
#                 if logdet is not None:
#                     logdet = logdet + logs.sum(dim=1)
#             else:
#                 t = self.f(y1, x)
#                 y2 = y2 + t
#
#             z = torch.cat((y1, y2), dim=1)
#             return z, x, logdet
#         else:
#             raise NotImplementedError('Inverse not implemented yet.')


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
