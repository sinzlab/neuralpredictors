from mlutils.general import flatten_json

def test_with_keep_nested_name():
    nested_dictionary = dict(a=0, dict2 = dict(b=0, dict3=dict(c=0)))
    flat_dictionary = dict(a=0, dict2_b=0, dict2_dict3_c=0)
    assert flatten_json(nested_dictionary, keep_nested_name=True) == flat_dictionary

def test_without_keep_nested_name():
    nested_dictionary = dict(a=0, dict2 = dict(b=0, dict3=dict(c=0)))
    flat_dictionary = dict(a=0, b=0, c=0)
    assert flatten_json(nested_dictionary, keep_nested_name=False) == flat_dictionary

