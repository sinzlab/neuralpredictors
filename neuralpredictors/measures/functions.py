import logging

logger = logging.getLogger(__name__)


def corr(y1, y2, dim=-1, eps=1e-8, **kwargs):
    """
    Compute the correlation between two PyTorch tensors along the specified dimension(s).

    Args:
        y1:      first PyTorch tensor
        y2:      second PyTorch tensor
        dim:     dimension(s) along which the correlation is computed. Any valid PyTorch dim spec works here
        eps:     offset to the standard deviation to avoid exploding the correlation due to small division (default 1e-8)
        **kwargs: passed to final numpy.mean operatoin over standardized y1 * y2

    Returns: correlation tensor
    """
    y1 = (y1 - y1.mean(dim=dim, keepdim=True)) / (y1.std(dim=dim, keepdim=True) + eps)
    y2 = (y2 - y2.mean(dim=dim, keepdim=True)) / (y2.std(dim=dim, keepdim=True) + eps)
    return (y1 * y2).mean(dim=dim, **kwargs)
