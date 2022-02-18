import warnings
from contextlib import contextmanager

import torch


@contextmanager
def eval_state(model):
    """
    Context manager, within which the model will be under `eval` mode.
    Upon existing, the model will return to whatever training state it
    was as it entered into the context.

    Args:
        model (PyTorch Module): PyTorch Module whose train/eval state is to be managed.

    Yields:
        PyTorch Module: The model switched to eval state.
    """
    training_status = model.training

    try:
        model.eval()
        yield model
    finally:
        model.train(training_status)


@contextmanager
def device_state(model, device):
    """
    Within the context, attemps to place the `model` onto the specified
    `device`. If `device` is CUDA and the specified device does not exist,
    the context falls back to using `cpu`. Upon existing the context, the model
    will be placed back on to the original device inferred based on the first entry
    of the model's parameter.

    Args:
        model (PyTorch Module): PyTorch Module object to swtich device.
        device (Any): target device descriptor. Any valid PyTorch device descriptor may be used.

    Yields:
        PyTorch Module: Model placed on the new device
    """
    # infer the original device based on the device the first parameter is placed on
    original_device = next(model.parameters()).device

    # create device spec
    # if device is simply "cuda", then device.index will evaluate to one, and the if statement will error out.
    device = torch.device("cuda:0") if device == "cuda" else torch.device(device)
    if device.type == "cuda" and device.index >= torch.cuda.device_count():
        # fall back to using CPU
        warnings.warn("Incompatible CUDA spec. Falling back to CPU usage")
        device = "cpu"

    try:
        model.to(device)
        yield model
    finally:
        model.to(original_device)
