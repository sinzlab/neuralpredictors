import logging
import warnings
from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from ..utils import anscombe

logger = logging.getLogger(__name__)


def corr(
    y1: ArrayLike, y2: ArrayLike, axis: Union[None, int, Tuple[int]] = -1, eps: int = 1e-8, **kwargs
) -> np.ndarray:
    """
    Compute the correlation between two NumPy arrays along the specified dimension(s).

    Args:
        y1:      first NumPy array
        y2:      second NumPy array
        axis:    dimension(s) along which the correlation is computed. Any valid NumPy axis spec works here
        eps:     offset to the standard deviation to avoid exploding the correlation due to small division (default 1e-8)
        **kwargs: passed to final numpy.mean operation over standardized y1 * y2

    Returns: correlation array
    """

    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=0) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=0) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)


def oracle_corr_conservative(repeated_outputs: ArrayLike) -> np.ndarray:
    """
    Compute the corrected oracle correlations per neuron.
    Note that an unequal number of repeats will introduce bias as it distorts assumptions made about the dataset.
    Note that oracle_corr_conservative overestimates the true oracle correlation.

    Args:
        repeated_outputs (array-like): numpy array with shape (images, repeats, neuron responses), or a list containing for each
            image an array of shape (repeats, neuron responses).

    Returns:
        array: Corrected oracle correlations per neuron
    """

    var_noise, var_mean = [], []
    for output in repeated_outputs:
        var_noise.append(output.var(axis=0))
        var_mean.append(output.mean(axis=0))
    var_noise = np.mean(np.array(var_noise), axis=0)
    var_mean = np.var(np.array(var_mean), axis=0)
    return var_mean / np.sqrt(var_mean * (var_mean + var_noise))


def oracle_corr_jackknife(repeated_outputs: ArrayLike) -> np.ndarray:
    """
    Compute the oracle correlations per neuron.
    Note that an unequal number of repeats will introduce bias as it distorts assumptions made about the dataset.
    Note that oracle_corr_jackknife underestimates the true oracle correlation.

    Args:
        repeated_outputs (array-like): numpy array with shape (images, repeats, neuron responses), or a list containing for each
            image an array of shape (repeats, neuron responses).

    Returns:
        array: Oracle correlations per neuron
    """

    oracles = []
    for outputs in repeated_outputs:
        num_repeats = outputs.shape[0]
        oracle = (outputs.sum(axis=0, keepdims=True) - outputs) / (num_repeats - 1)
        if np.any(np.isnan(oracle)):
            logger.warning(
                "{}% NaNs when calculating the oracle. NaNs will be set to Zero.".format(np.isnan(oracle).mean() * 100)
            )
            oracle[np.isnan(oracle)] = 0
        oracles.append(oracle)
    return corr(np.vstack(repeated_outputs), np.vstack(oracles), axis=0)


def explainable_var(repeated_outputs: ArrayLike, eps: int = 1e-9) -> np.ndarray:
    """
    Compute the explainable variance per neuron.

    Args:
        repeated_outputs (array): numpy array with shape (images, repeats, neuron responses), or a list containing for each
            image an array of shape (repeats, neuron responses).

    Returns:
        array: Corrected oracle correlations per neuron
    """

    total_var = np.var(np.vstack(repeated_outputs), axis=0, ddof=1)
    img_var = np.var(repeated_outputs, axis=1, ddof=1)
    noise_var = np.mean(img_var, axis=0)
    explainable_var = (total_var - noise_var) / (total_var + eps)
    return explainable_var


def fev(targets: ArrayLike, predictions: ArrayLike, return_exp_var: bool = False) -> Union[ArrayLike, Tuple[ArrayLike]]:
    """
    Compute the fraction of explainable variance explained per neuron

    Args:
        targets (array-like): Neuronal neuron responses (ground truth) to image repeats. Dimensions:
            [num_images] np.array(num_repeats, num_neurons)
        outputs (array-like): Model predictions to the repeated images, with an identical shape as the targets
        return_exp_var (bool): returns the fraction of explainable variance per neuron if set to True
    Returns:
        FEVe (np.array): the fraction of explainable variance explained per neuron
        --- optional: FEV (np.array): the fraction
    """

    img_var = []
    pred_var = []
    for target, prediction in zip(targets, predictions):
        pred_var.append((target - prediction) ** 2)
        img_var.append(np.var(target, axis=0, ddof=1))
    pred_var = np.vstack(pred_var)
    img_var = np.vstack(img_var)

    total_var = np.var(np.vstack(targets), axis=0, ddof=1)
    noise_var = np.mean(img_var, axis=0)
    fev = (total_var - noise_var) / total_var

    pred_var = np.mean(pred_var, axis=0)
    fev_e = 1 - (pred_var - noise_var) / (total_var - noise_var)
    return [fev, fev_e] if return_exp_var else fev_e


def snr(repeated_outputs: ArrayLike, per_neuron: bool = True) -> np.ndarray:
    """
    Compute signal to noise ratio.

    Args:
        repeated_outputs (array-like): numpy array with shape (images, repeats, neuron responses), or a list containing for each
            image an array of shape (repeats, neuron responses).
        per_neuron (bool, optional): Return snr per neuron or averaged across neurons. Defaults to True.

    Returns:
        array: Signal to noise ratio per neuron or averaged across neurons.
    """
    repeated_outputs = anscombe(repeated_outputs)
    mu = np.array([np.mean(repeats, axis=0) for repeats in repeated_outputs])
    mu_bar = np.mean(mu, axis=0)
    sigma_2 = np.array([np.var(repeats, ddof=1, axis=0) for repeats in repeated_outputs])
    sigma_2_bar = np.mean(sigma_2, axis=0)
    snr = (1 / mu.shape[0] * np.sum((mu - mu_bar) ** 2, axis=0)) / sigma_2_bar
    return snr if per_neuron else np.mean(snr)


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
