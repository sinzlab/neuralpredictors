import warnings
import numpy as np
import h5py
import torch
from torch import nn as nn
from torch.nn import Parameter


def flatten_json(nested_dict, keep_nested_name=True):
    """Turns a nested dictionary into a flattened dictionary. Designed to facilitate the populating of Config.Part tables
    with the corresponding config json list of parameters from the Config master table.

    Args:
        nested_dict: dict
            Nested dictionary to be flattened
        keep_nested_name: boolean, default True
            If True, record names will consist of all nested names separated by '_'. If False, last record name is
            chosen as new recod name. This is only possible for unique record names.

    Returns: dict
            Flattened dictionary

    Raises:
        ValueError: Multiple entries with identical names
    """
    out = {}

    def flatten(x, name=""):
        if isinstance(x, dict):
            for key, value in x.items():
                flatten(value, (name if keep_nested_name else "") + key + "_")
        else:
            if name[:-1] in out:
                raise ValueError("Multiple entries with identical names")
            out[name[:-1]] = x

    flatten(nested_dict)
    return out


def gini(x):
    """ Calculates the Gini coefficient from a list of numbers. The Gini coefficient is used as a measure of (in)equality
    where a Gini coefficient of 1 (or 100%) expresses maximal inequality among values. A value greater than one may occur
     if some value represents negative contribution.

    Args:
        x: 1 D array or list
            Array of numbers from which to calculate the Gini coefficient.

    Returns: float
            Gini coefficient

    """
    x = np.asarray(x)  # The code below requires numpy arrays.
    if any(i < 0 for i in x):
        warnings.warn("Input x contains negative values")
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def load_dict_from_hdf5(filename):
    """
    Given a `filename` of a HDF5 file, opens the file and
    load the entire content as a (nested) dictionary.

    Args:
        filename - name of HDF5 file

    Returns:
        (nested) dictionary corresponding to the content of the HDF5 file.
    """
    with h5py.File(filename, "r") as h5file:
        return recursively_load_dict_contents_from_group(h5file)


def recursively_load_dict_contents_from_group(h5file, path="/"):
    """
    Given a `h5file` h5py object, loads the object at `path`
    as nested dictionary.

    Args:
        h5file - h5py object
        path - Path within the h5py file to load the content of recursively.

    Returns:
        (nested) dictionary corresponding to the content of the HDF5 file at the path.
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            dtype = item.dtype
            v = item[()]
            if dtype.char == "S":  # convert bytes to univcode
                v = v.astype(str)
            ans[key] = v
        elif isinstance(item, h5py.Group):
            if item.attrs.get("_iterable", False):
                ans[key] = [item[str(i)][()] for i in range(len(item))]
            else:
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + "/")
    return ans


class BiasNet(nn.Module):
    """
    Small helper network that adds a learnable bias to an already instantiated base network
    """

    def __init__(self, base_net):
        super(BiasNet, self).__init__()
        self.bias = Parameter(torch.Tensor(2))
        self.base_net = base_net

    def forward(self, x):
        return self.base_net(x) + self.bias
