import warnings
import numpy as np
from scipy import stats

def compute_predictions(loader, model):
    y, y_hat = [], []
    for x_val, y_val in loader:
        y_hat.append(model(x_val).detach().cpu().numpy())
        y.append(y_val.detach().cpu().numpy())
    y, y_hat = map(np.vstack, (y, y_hat))
    return y, y_hat


def corr_stop(model):
    with eval_state(model):
        y, y_hat = compute_predictions(val_loader, model)

    ret = corr(y, y_hat, axis=0)

    if np.any(np.isnan(ret)):
        warnings.warn('{}% NaNs '.format(np.isnan(ret).mean() * 100))
    ret[np.isnan(ret)] = 0

    return ret.mean()


def gamma_stop(model):
    with eval_state(model):
        y, y_hat = compute_predictions(val_loader, model)

    ret = -stats.gamma.logpdf(y + 1e-7, y_hat + 0.5).mean(axis=1) / np.log(2)
    if np.any(np.isnan(ret)):
        warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
    ret[np.isnan(ret)] = 0
    # -- average if requested
    return ret.mean()


def exp_stop(model, bias=1e-12, target_bias=1e-7):
    with eval_state(model):
        y, y_hat = compute_predictions(val_loader, model)
    y = y + target_bias
    y_hat = y_hat + bias
    ret = (y / y_hat + np.log(y_hat)).mean(axis=1) / np.log(2)
    if np.any(np.isnan(ret)):
        warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
    ret[np.isnan(ret)] = 0
    # -- average if requested
    return ret.mean()


def poisson_stop(model):
    with eval_state(model):
        target, output = compute_predictions(val_loader, model)

    ret = (output - target * np.log(output + 1e-12))
    if np.any(np.isnan(ret)):
        warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
    ret[np.isnan(ret)] = 0
    # -- average if requested
    return ret.mean()
