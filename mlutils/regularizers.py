import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# def laplace():
#     return np.array([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]).astype(np.float32)[None, None, ...]


def laplace():
    """
    Returns a 3x3 laplace filter.

    """
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)[None, None, ...]


def laplace3d():
    l = np.zeros((3, 3, 3))
    l[1, 1, 1] = -6.0
    l[1, 1, 2] = 1.0
    l[1, 1, 0] = 1.0
    l[1, 0, 1] = 1.0
    l[1, 2, 1] = 1.0
    l[0, 1, 1] = 1.0
    l[2, 1, 1] = 1.0
    return l.astype(np.float32)[None, None, ...]


def gaussian2d(size, sigma=5, gamma=1, theta=0, center=(0, 0), normalize=True):
    """
    Returns a 2D Gaussian filter.

    Args:
        size (tuple of int, or int): Image height and width.
        sigma (float): std deviation of the Gaussian along x-axis. Default is 5..
        gamma (float): ratio between std devidation along x-axis and y-axis. Default is 1.
        theta (float): Orientation of the Gaussian (in ratian). Default is 0.
        center (tuple): The position of the filter. Default is center (0, 0).
        normalize (bool): Whether to normalize the entries. This is computed by
            subtracting the minimum value and then dividing by the max. Default is True.

    Returns:
        2D Numpy array: A 2D Gaussian filter.

    """

    sigma_x = sigma
    sigma_y = sigma / gamma

    xmax, ymax = (size, size) if isinstance(size, int) else size
    xmax, ymax = (xmax - 1) / 2, (ymax - 1) / 2
    xmin, ymin = -xmax, -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # shift the position
    y -= center[0]
    x -= center[1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gaussian = np.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))

    if normalize:
        gaussian -= gaussian.min()
        gaussian /= gaussian.max()

    return gaussian.astype(np.float32)


class Laplace(nn.Module):
    """
    Laplace filter for a stack of data. Utilized as the input weight regularizer.
    """

    def __init__(self, padding=None):
        """
        Laplace filter for a stack of data.

        Args:
            padding (int): Controls the amount of zero-padding for the convolution operation
                            Default is half of the kernel size (recommended)

        Attributes:
            filter (2D Numpy array): 3x3 Laplace filter.
            padding_size (int): Number of zeros added to each side of the input image
                before convolution operation.
        """
        super().__init__()
        self.register_buffer("filter", torch.from_numpy(laplace()))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def forward(self, x):
        return F.conv2d(x, self.filter, bias=None, padding=self.padding_size)


class LaplaceL2(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer. Unnormalized, not recommended to use.
        Use LaplaceL2norm instead.

        Args:
            padding (int): Controls the amount of zero-padding for the convolution operation.

        Attributes:
            laplace (Laplace): Laplace convolution object. The output is the result of
                convolving an input image with laplace filter.

    """

    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace(padding=padding)
        warnings.warn("LaplaceL2 Regularizer is deprecated. Use LaplaceL2norm instead.")

    def forward(self, x, avg=True):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1, k2)).pow(2)) / 2


class LaplaceL2norm(nn.Module):
    """
    Normalized Laplace regularizer for a 2D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1, k2)).pow(2)) / agg_fn(x.view(oc * ic, 1, k1, k2).pow(2))


class Laplace3d(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self, padding=None):
        super().__init__()
        self.register_buffer("filter", torch.from_numpy(laplace3d()))

    def forward(self, x):
        return F.conv3d(x, self.filter, bias=None)


class LaplaceL23d(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace3d()

    def forward(self, x):
        oc, ic, k1, k2, k3 = x.size()
        return self.laplace(x.view(oc * ic, 1, k1, k2, k3)).pow(2).mean() / 2


class FlatLaplaceL23d(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, x):
        oc, ic, k1, k2, k3 = x.size()
        assert k1 == 1, "time dimension must be one"
        return self.laplace(x.view(oc * ic, 1, k2, k3)).pow(2).mean() / 2


class LaplaceL1(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self, padding=0):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, avg=True):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.view(oc * ic, 1, k1, k2)).abs())


class GaussianLaplaceL2Adaptive(nn.Module):
    """
    Laplace regularizer, with a Gaussian mask, for a 2D convolutional layer.
        Is flexible across kernel sizes, so that the regularizer can be applied to more
        than one layer, with changing kernel sizes
    """

    def __init__(self, padding=None, sigma=None):
        """
        Args:
            padding (int): Controls the amount of zero-padding for the convolution operation.
            sigma (float): std deviation of the Gaussian along x-axis. Default is calculated
                as the 1/4th of the minimum dimenison (height vs width) of the input.
        """
        super().__init__()
        self.laplace = Laplace(padding=padding)
        self.sigma = sigma

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        sigma = self.sigma if self.sigma else min(k1, k2) / 4

        out = self.laplace(x.view(ic * oc, 1, k1, k2))
        out = out * (1 - torch.from_numpy(gaussian2d(size=(k1, k2), sigma=sigma)).expand(1, 1, k1, k2).to(x.device))

        return agg_fn(out.pow(2)) / agg_fn(x.view(oc * ic, 1, k1, k2).pow(2))


class GaussianLaplaceL2(nn.Module):
    """
    Laplace regularizer, with a Gaussian mask, for a single 2D convolutional layer.

    """

    def __init__(self, kernel, padding=None):
        """
        Args:
            kernel: Size of the convolutional kernel of the filter that is getting regularized
            padding (int): Controls the amount of zero-padding for the convolution operation.
        """
        super().__init__()

        self.laplace = Laplace(padding=padding)
        self.kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        sigma = min(*self.kernel) / 4
        self.gaussian2d = torch.from_numpy(gaussian2d(size=(*self.kernel,), sigma=sigma))

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        out = self.laplace(x.view(oc * ic, 1, k1, k2))
        out = out * (1 - self.gaussian2d.expand(1, 1, k1, k2).to(x.device))

        return agg_fn(out.pow(2)) / agg_fn(x.view(oc * ic, 1, k1, k2).pow(2))
