import torch
import numpy as np


def fitted_zig_mean(theta, k, loc, q, approximate=False):
    uniform_mean = 0
    if not approximate:
        uniform_mean += (1 - q) * 0.5 * loc
    gamma_mean = q * (k * theta + loc)
    return uniform_mean + gamma_mean


def fitted_zig_variance(theta, k, loc, q, approximate=False):
    expectation_of_variances = q * k * theta**2
    if not approximate:
        expectation_of_variances += (1 - q) * loc**2 / 12

    variance_of_expectations = q * (1 - q) * (k * theta + loc / 2) ** 2
    return expectation_of_variances + variance_of_expectations


def fitted_zil_mean(mu, sigma2, q, loc, use_torch=True):
    exp = torch.exp if use_torch else np.exp
    A = q * (exp(mu + sigma2 / 2) + loc)

    # Set NaN to 0
    nan_idx = torch.where(torch.isnan(A))[0] if use_torch else np.where(np.isnan(A))[0]
    A[nan_idx] = 0
    return (1 - q) * loc / 2 + A


def fitted_zil_variance(mu, sigma2, q, loc, use_torch=True):
    exp = torch.exp if use_torch else np.exp
    A = q * (exp(sigma2) - 1) * exp(2 * mu + sigma2)
    B = q * (1 - q) * (loc / 2 - exp(mu + sigma2 / 2) - loc) ** 2

    nan_idx = torch.where(torch.isnan(A))[0] if use_torch else np.where(np.isnan(A))[0]
    A[nan_idx] = 0
    B[nan_idx] = 0
    return (1 - q) * loc**2 / 12 + A + B