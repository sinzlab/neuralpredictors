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
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)

    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
