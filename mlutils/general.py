def flatten_json(nested_dict, keep_nested_name=True):
    '''
    Turns a nested dictionary into a flattened dictionary. Designed to facilitate the populating of Config.Part tables
    with the corresponding config json list of parameters from the Config master table.
    Args:
        nested_dict: dict
            Nested dictionary to be flattened
        keep_nested_name: boolean, default True
            If True, record names will consist of all nested names separated by '_'. If False, last record name is
            chosen as new recod name. This is only possible for unique record names.

    Returns: dict
            Flattened dictionary
    '''
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], (name if keep_nested_name else '') + a + '_')
        else:
            if name[:-1] in out: raise ValueError('Multiple entries with identical names')
            out[name[:-1]] = x

    flatten(nested_dict)
    return out