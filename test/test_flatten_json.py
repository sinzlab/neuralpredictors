from neuralpredictors.utils import flatten_json
import pytest


def nested_dictionary(duplicates):
    dictionary = dict(a=0, dict2=dict(b=0, dict3=dict(c=0)))
    if duplicates:
        dictionary["c"] = 0
    return dictionary


def flat_dictionary(keep_nested_name):
    if keep_nested_name:
        return dict(a=0, dict2_b=0, dict2_dict3_c=0)
    else:
        return dict(a=0, b=0, c=0)


@pytest.mark.parametrize("keep_nested_name", [True, False])
def test_output(keep_nested_name):
    assert flatten_json(nested_dictionary(duplicates=False), keep_nested_name) == flat_dictionary(keep_nested_name)


def test_exception():
    with pytest.raises(ValueError):
        flatten_json(nested_dictionary(duplicates=True), keep_nested_name=False)
