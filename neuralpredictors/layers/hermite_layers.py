import math
import numpy as np
from numpy import pi
from scipy.special import gamma
from numpy.polynomial.polynomial import polyval


import torch
from torch import nn


def hermite_coefficients(mu, nu):
    """Generate coefficients of 2D Hermite functions"""
    nur = np.arange(nu + 1)
    num = gamma(mu + nu + 1) * gamma(nu + 1) * ((-2) ** (nu - nur))
    denom = gamma(mu + 1 + nur) * gamma(1 + nur) * gamma(nu + 1 - nur)
    return num / denom


def hermite_2d(N, npts, xvalmax=None):
    """Generate 2D Hermite function basis

    Arguments:
    N           -- the maximum rank.
    npts        -- the number of points in x and y

    Keyword arguments:
    xvalmax     -- the maximum x and y value (default: 2.5 * sqrt(N))

    Returns:
    H           -- Basis set of size N*(N+1)/2 x npts x npts
    desc        -- List of descriptors specifying for each
                   basis function whether it is:
                        'z': rotationally symmetric
                        'r': real part of quadrature pair
                        'i': imaginary part of quadrature pair

    """
    xvalmax = xvalmax or 2.5 * np.sqrt(N)
    ranks = range(N)

    # Gaussian envelope
    xvalmax *= 1 - 1 / npts
    xvals = np.linspace(-xvalmax, xvalmax, npts, endpoint=True)[..., None]

    gxv = np.exp(-xvals ** 2 / 4)
    gaussian = np.dot(gxv, gxv.T)

    # Hermite polynomials
    mu = np.array([])
    nu = np.array([])
    desc = []
    for i, rank in enumerate(ranks):
        muadd = np.sort(np.abs(np.arange(-rank, rank + 0.1, 2)))
        mu = np.hstack([mu, muadd])
        nu = np.hstack([nu, (rank - muadd) / 2])
        if not (rank % 2):
            desc.append("z")
        desc += ["r", "i"] * int(np.floor((rank + 1) / 2))

    theta = np.arctan2(xvals, xvals.T)
    radsq = xvals ** 2 + xvals.T ** 2
    nbases = mu.size
    H = np.zeros([nbases, npts, npts])
    for i, (mui, nui, desci) in enumerate(zip(mu, nu, desc)):
        radvals = polyval(radsq, hermite_coefficients(mui, nui))
        basis = gaussian * (radsq ** (mui / 2)) * radvals * np.exp(1j * mui * theta)
        basis /= np.sqrt(2 ** (mui + 2 * nui) * pi * math.factorial(mui + nui) * math.factorial(nui))
        if desci == "z":
            H[i] = basis.real / np.sqrt(2)
        elif desci == "r":
            H[i] = basis.real
        elif desci == "i":
            H[i] = basis.imag

    # normalize
    return H / np.sqrt(np.sum(H ** 2, axis=(1, 2), keepdims=True)), desc, mu


def rotation_matrix(desc, mu, angle):
    R = np.zeros((len(desc), len(desc)))
    for i, (d, m) in enumerate(zip(desc, mu)):
        if d == "r":
            Rc = np.array([[np.cos(m * angle), np.sin(m * angle)], [-np.sin(m * angle), np.cos(m * angle)]])
            R[i : i + 2, i : i + 2] = Rc
        elif d == "z":
            R[i, i] = 1
    return R


def downsample_weights(weights, factor=2):
    w = 0
    for i in range(factor):
        for j in range(factor):
            w += weights[i::factor, j::factor]
    return w


class RotateHermite(nn.Module):
    def __init__(self, filter_size, upsampling, num_rotations, first_layer):

        super().__init__()

        H, desc, mu = hermite_2d(filter_size, filter_size * upsampling, 2 * np.sqrt(filter_size))

        self.H = nn.Parameter(torch.tensor(H, dtype=torch.float32), requires_grad=False)

        angles = [i * 2 * pi / num_rotations for i in range(num_rotations)]
        Rs = [torch.tensor(rotation_matrix(desc, mu, angle), dtype=torch.float32) for angle in angles]

        self.Rs = nn.ParameterList([nn.Parameter(R, requires_grad=False) for R in Rs])

        self.num_rotations = num_rotations
        self.first_layer = first_layer

    def forward(self, coeffs):
        num_coeffs, num_inputs_total, num_outputs = coeffs.shape
        filter_size = self.H.shape[1]
        num_inputs = num_inputs_total // self.num_rotations
        weights_rotated = []
        for i, R in enumerate(self.Rs):
            coeffs_rotated = torch.tensordot(R, coeffs, dims=([1], [0]))
            w = torch.tensordot(self.H, coeffs_rotated, dims=[[0], [0]])
            if i and not self.first_layer:
                shift = num_inputs_total - i * num_inputs
                w = torch.cat([w[:, :, shift:, :], w[:, :, :shift, :]], dim=2)
            weights_rotated.append(w)
        weights_all_rotations = torch.cat(weights_rotated, dim=3)
        return weights_all_rotations


class HermiteConv2D(nn.Module):
    def __init__(
        self, input_features, output_features, filter_size, padding, stride, num_rotations, upsampling, first_layer
    ):

        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.padding = padding
        self.stride = stride
        self.upsampling = upsampling
        self.n_coeffs = filter_size * (filter_size + 1) // 2

        coeffs = nn.Parameter(torch.Tensor(self.n_coeffs, self.input_features, self.output_features))
        self.coeffs = coeffs

        self.rotate_hermite = RotateHermite(
            filter_size=filter_size, upsampling=upsampling, num_rotations=num_rotations, first_layer=first_layer
        )

        self.weights_all_rotations = None

    def forward(self, input):
        weights_all_rotations = self.rotate_hermite(self.coeffs)
        weights_all_rotations = downsample_weights(weights_all_rotations, self.upsampling)
        weights_all_rotations = weights_all_rotations.permute(3, 2, 0, 1)
        self.weights_all_rotations = weights_all_rotations

        return nn.functional.conv2d(
            input=input, weight=self.weights_all_rotations, bias=None, stride=self.stride, padding=self.padding
        )


class RotationEquivariantBatchNorm2D(nn.Module):
    def __init__(self, num_features, num_rotations, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):

        super().__init__()

        self.num_features = num_features
        self.num_rotations = num_rotations
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.batch_norm = nn.BatchNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, input):
        s = input.shape
        input = self.reshape(input, s)
        output = self.batch_norm(input)
        output = self.inv_reshape(output, s)
        return output

    def reshape(self, x, s):
        x = x.view(s[0], self.num_rotations, self.num_features, s[2], s[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(s[0], self.num_features, self.num_rotations * s[2] * s[3])
        return x

    def inv_reshape(self, x, s):
        x = x.view(s[0], self.num_features, self.num_rotations, s[2], s[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(s[0], self.num_rotations * self.num_features, s[2], s[3])
        return x


class RotationEquivariantBias2DLayer(nn.Module):
    def __init__(self, channels, num_rotations, initial=0, **kwargs):
        super().__init__(**kwargs)

        self.num_features = channels
        self.num_rotations = num_rotations
        self.bias = torch.nn.Parameter(torch.empty((1, 1, channels, 1, 1)).fill_(initial))

    def forward(self, x):
        s = x.shape
        x = x.view(s[0], self.num_rotations, self.num_features, s[2], s[3])
        x = x + self.bias
        return x.view(*s)


class RotationEquivariantScale2DLayer(nn.Module):
    def __init__(self, channels, num_rotations, initial=1, **kwargs):
        super().__init__(**kwargs)

        self.num_features = channels
        self.num_rotations = num_rotations
        self.scale = torch.nn.Parameter(torch.empty((1, 1, channels, 1, 1)).fill_(initial))

    def forward(self, x):
        s = x.shape
        x = x.view(s[0], self.num_rotations, self.num_features, s[2], s[3])
        x = x * self.scale
        return x.view(*s)
