import collections
from typing import TypeVar

T = TypeVar("T", bound=collections.abc.Mapping)


def deep_update(d: T, u: collections.abc.Mapping) -> T:
    """
    Recursively update a dictionary with the values from another dictionary
    Args:
        d: dictionary to be updated
        u: dictionary that is used to update `d`

    Returns:
        updated dictionary `d`
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
