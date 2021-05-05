import pytest

from src.policies.policy_tools import combine_dictionaries
from src.policies.policy_tools import filter_dictionary
from src.policies.policy_tools import update_dictionary


def test_filter_dictionary():
    d = {"a": 0, "b": 1, "c": 2}
    res = filter_dictionary(lambda x: x in ["a", "b"], d)
    expected = {"a": 0, "b": 1}
    assert res == expected


def test_filter_dictionary_two():
    d = {0: "a", 1: "b", 2: "c"}
    res = filter_dictionary(lambda x: x > 1, d)
    expected = {2: "c"}
    assert res == expected


def test_filter_dictionary_three():
    d = {0: "a", 1: "b", 2: "c"}
    res = filter_dictionary(lambda x: x == "c", d, by="values")
    expected = {2: "c"}
    assert res == expected


def test_update_dictionary():
    d1 = {"a": 0, "b": 2}
    to_add = {"c": 3}
    res = update_dictionary(d1, to_add)
    assert d1 == {"a": 0, "b": 2}
    assert to_add == {"c": 3}
    assert res == {"a": 0, "b": 2, "c": 3}


def test_combine_dictionaries_unproblematic():
    d1 = {"a": 0}
    d2 = {"b": 1}
    d3 = {"c": 2}
    res = combine_dictionaries([d1, d2, d3])
    assert d1 == {"a": 0}
    assert d2 == {"b": 1}
    assert d3 == {"c": 2}
    assert res == {"a": 0, "b": 1, "c": 2}


def test_combine_dictionaries_duplicate():
    d1 = {"a": 0}
    d2 = {"b": 1}
    d3 = {"a": 2}
    with pytest.raises(ValueError):
        combine_dictionaries([d1, d2, d3])
