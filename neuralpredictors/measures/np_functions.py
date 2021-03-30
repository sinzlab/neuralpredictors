import logging
import numpy as np
import warnings

logger = logging.getLogger(__name__)


def corr(y1, y2, axis=-1, eps=1e-8, **kwargs):
    """
    Compute the correlation between two NumPy arrays along the specified dimension(s).

    Args:
        y1:      first NumPy array
        y2:      second NumPy array
        axis:    dimension(s) along which the correlation is computed. Any valid NumPy axis spec works here
        eps:     offset to the standard deviation to avoid exploding the correlation due to small division (default 1e-8)
        **kwargs: passed to final numpy.mean operatoin over standardized y1 * y2

    Returns: correlation array
    """
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=0) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=0) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)


def gini(x, axis=None):
    """
    Calculate the Gini coefficient from a list of numbers. The Gini coefficient is used as a measure of (in)equality
    where a Gini coefficient of 1 (or 100%) expresses maximal inequality among values. A value greater than 1 may occur
     if some value represents negative contribution.

    Args:
        x: 1 D array or list
            Array of numbers from which to calculate the Gini coefficient.
        axis: axis along which to compute gini. If None, then the array is flattened first.

    Returns: float
            Gini coefficient
    """
    x = np.asarray(x)
    if axis is None:
        x = x.flatten()
        axis = -1
    if np.any(x < 0):
        warnings.warn("Input x contains negative values")
    sorted_x = np.sort(x, axis=axis)
    n = x.shape[axis]
    cumx = np.cumsum(sorted_x, dtype=float, axis=axis)
    return (n + 1 - 2 * np.sum(cumx, axis=axis) / cumx.take(-1, axis=axis)) / n

