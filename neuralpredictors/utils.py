from contextlib import contextmanager

import numpy as np
import torch

from .training import eval_state


def get_module_output(model, input_shape, use_cuda=True):
    """
    Return the output shape of the model when fed in an array of `input_shape`.
    Note that a zero array of shape `input_shape` is fed into the model and the
    shape of the output of the model is returned.

    Args:
        model (nn.Module): PyTorch module for which to compute the output shape
        input_shape (tuple): Shape specification for the input array into the model
        use_cuda (bool, optional): If True, model will be evaluated on CUDA if available. Othewrise
            model evaluation will take place on CPU. Defaults to True.

    Returns:
        tuple: output shape of the model

    """
    # infer the original device
    initial_device = next(iter(model.parameters())).device
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    with eval_state(model):
        with torch.no_grad():
            input = torch.zeros(1, *input_shape[1:], device=device)
            output = model.to(device)(input)
    model.to(initial_device)
    return output.shape


@contextmanager
def no_transforms(dat):
    """
    Contextmanager for the dataset object. It temporarily removes the transforms.
    Args:
        dat: Dataset object. Either FileTreeDataset or StaticImageSet

    Yields: The dataset object without transforms
    """
    transforms = dat.transforms
    try:
        dat.transforms = []
        yield dat
    finally:
        dat.transforms = transforms


def anscombe(x):
    """Compute Anscombe transform."""
    return 2 * np.sqrt(x + 3 / 8)
