import logging

import numpy as np

logger = logging.getLogger(__name__)


def corr(y1, y2, axis=-1, eps=1e-8, **kwargs):
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


def oracle_corr_corrected(repeated_outputs):
    """
    Compute the corrected oracle correlations per neuron.

    Args:
        repeated_outputs (array-like): numpy array with shape (images, repeats, responses), or a list of lists of
            responses (with len: num of repeats) per image.

    Returns:
        array: Corrected oracle correlations per neuron
    """

    if len(repeated_outputs.shape) == 3:
        var_noise = repeated_outputs.var(axis=1).mean(0)
        var_mean = repeated_outputs.mean(axis=1).var(0)
    else:
        var_noise, var_mean = [], []
        for repeat in repeated_outputs:
            var_noise.append(repeat.var(axis=0))
            var_mean.append(repeat.mean(axis=0))
        var_noise = np.mean(np.array(var_noise), axis=0)
        var_mean = np.var(np.array(var_mean), axis=0)
    return var_mean / np.sqrt(var_mean * (var_mean + var_noise))


def oracle_corr(repeated_outputs):
    """
    Compute the oracle correlations per neuron.

    Args:
        repeated_outputs (array-like): numpy array with shape (images, repeats, responses), or a list of lists of
            responses (with len: num of repeats) per image.

    Returns:
        array: Oracle correlations per neuron
    """

    if len(repeated_outputs.shape) == 3:
        _, r, n = repeated_outputs.shape
        oracles = (repeated_outputs.mean(axis=1, keepdims=True) - repeated_outputs / r) * r / (r - 1)
        if np.any(np.isnan(oracles)):
            logger.warning(
                "{}% NaNs when calculating the oracle. NaNs will be set to Zero.".format(np.isnan(oracles).mean() * 100)
            )
        oracles[np.isnan(oracles)] = 0
        return corr(oracles.reshape(-1, n), repeated_outputs.reshape(-1, n), axis=0)
    else:
        oracles = []
        for outputs in repeated_outputs:
            r, n = outputs.shape
            # compute the mean over repeats, for each neuron
            mu = outputs.mean(axis=0, keepdims=True)
            # compute oracle predictor
            oracle = (mu - outputs / r) * r / (r - 1)

            if np.any(np.isnan(oracle)):
                logger.warning(
                    "{}% NaNs when calculating the oracle. NaNs will be set to Zero.".format(
                        np.isnan(oracle).mean() * 100
                    )
                )
                oracle[np.isnan(oracle)] = 0

            oracles.append(oracle)
        return corr(np.vstack(repeated_outputs), np.vstack(oracles), axis=0)


def explainable_var(repeated_outputs, eps=1e-9):
    """
    Compute the explainable variance per neuron.

    Args:
        repeated_outputs (array): numpy array with shape (images, repeats, responses), or a list of lists of
            responses (with len: num of repeats) per image.

    Returns:
        array: Corrected oracle correlations per neuron
    """

    ImgVariance = []
    TotalVar = np.var(np.vstack(repeated_outputs), axis=0, ddof=1)
    for out in repeated_outputs:
        ImgVariance.append(np.var(out, axis=0, ddof=1))
    ImgVariance = np.vstack(ImgVariance)
    NoiseVar = np.mean(ImgVariance, axis=0)
    explainable_var = (TotalVar - NoiseVar) / (TotalVar + eps)
    return explainable_var


def fev(targets, predictions, return_exp_var=False):
    """
    Compute the fraction of explainable variance explained per neuron

    Args:
        targets (list): Neuronal responses (ground truth) to image repeats. Dimensions:
            [num_images] np.array(num_repeats, num_neurons)
        outputs (list): Model predictions to the repeated images, with an identical shape as the targets
        return_exp_var (bool): returns the fraction of explainable variance per neuron if set to True
    Returns:
        FEVe (np.array): the fraction of explainable variance explained per neuron
        --- optional: FEV (np.array): the fraction
    """

    ImgVariance = []
    PredVariance = []
    for i, _ in enumerate(targets):
        PredVariance.append((targets[i] - predictions[i]) ** 2)
        ImgVariance.append(np.var(targets[i], axis=0, ddof=1))
    PredVariance = np.vstack(PredVariance)
    ImgVariance = np.vstack(ImgVariance)

    TotalVar = np.var(np.vstack(targets), axis=0, ddof=1)
    NoiseVar = np.mean(ImgVariance, axis=0)
    FEV = (TotalVar - NoiseVar) / TotalVar

    PredVar = np.mean(PredVariance, axis=0)
    FEVe = 1 - (PredVar - NoiseVar) / (TotalVar - NoiseVar)
    return [FEV, FEVe] if return_exp_var else FEVe


def snr(repeated_outputs, per_neuron=True):
    """
    Compute signal to noise ratio.

    Args:
        repeated_outputs (array-like): numpy array with shape (images, repeats, responses), or a list of lists of
            responses (with len: num of repeats) per image.
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


def anscombe(x):
    """Helper function for snr"""
    return 2 * np.sqrt(x + 3 / 8)
