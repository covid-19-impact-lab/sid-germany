import pandas as pd
import pytest

from src.policies.policy_tools import combine_dictionaries
from src.policies.policy_tools import filter_dictionary
from src.policies.policy_tools import remove_educ_policies
from src.policies.policy_tools import remove_other_policies
from src.policies.policy_tools import remove_school_policies
from src.policies.policy_tools import remove_work_policies
from src.policies.policy_tools import remove_young_educ_policies
from src.policies.policy_tools import shorten_policies
from src.policies.policy_tools import split_policies
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


@pytest.fixture
def policies():
    d = {
        "drop_bc_too_early": {
            "start": "2020-05-01",
            "end": "2020-05-30",
            "multiplier": 0.5,
        },
        "drop_bc_too_late": {
            "start": "2021-01-01",
            "end": "2021-01-30",
            "multiplier": 0.2,
        },
        "adjust_to_start_later": {
            "start": "2020-09-01",
            "end": "2020-11-01",
            "multiplier": 1.0,
        },
        "adjust_to_end_earlier": {
            "start": "2020-11-02",
            "end": "2020-12-30",
            "multiplier": 1.0,
        },
        "unchanged": {
            "start": "2020-11-02",
            "end": "2020-11-30",
            "multiplier": 1.0,
        },
    }
    return d


def test_shorten_policies_both_dates(policies):
    start = pd.Timestamp("2020-10-01")
    end = pd.Timestamp("2020-12-01")

    res = shorten_policies(policies, start_date=start, end_date=end)

    expected = {
        "adjust_to_start_later": {
            "start": start,
            "end": pd.Timestamp("2020-11-01"),
            "multiplier": 1.0,
        },
        "adjust_to_end_earlier": {
            "start": pd.Timestamp("2020-11-02"),
            "end": end,
            "multiplier": 1.0,
        },
        "unchanged": {
            "start": pd.Timestamp("2020-11-02"),
            "end": pd.Timestamp("2020-11-30"),
            "multiplier": 1.0,
        },
    }
    assert res == expected


def test_shorten_policies_start_only(policies):
    start = pd.Timestamp("2020-10-01")

    res = shorten_policies(policies, start_date=start)
    expected = {
        "drop_bc_too_late": {
            "start": pd.Timestamp("2021-01-01"),
            "end": pd.Timestamp("2021-01-30"),
            "multiplier": 0.2,
        },
        "adjust_to_start_later": {
            "start": start,
            "end": pd.Timestamp("2020-11-01"),
            "multiplier": 1.0,
        },
        "adjust_to_end_earlier": {
            "start": pd.Timestamp("2020-11-02"),
            "end": pd.Timestamp("2020-12-30"),
            "multiplier": 1.0,
        },
        "unchanged": {
            "start": pd.Timestamp("2020-11-02"),
            "end": pd.Timestamp("2020-11-30"),
            "multiplier": 1.0,
        },
    }
    assert res == expected


def test_shorten_policies_end_only(policies):
    end = pd.Timestamp("2020-12-01")

    res = shorten_policies(policies, end_date=end)
    expected = {
        "drop_bc_too_early": {
            "start": pd.Timestamp("2020-05-01"),
            "end": pd.Timestamp("2020-05-30"),
            "multiplier": 0.5,
        },
        "adjust_to_start_later": {
            "start": pd.Timestamp("2020-09-01"),
            "end": pd.Timestamp("2020-11-01"),
            "multiplier": 1.0,
        },
        "adjust_to_end_earlier": {
            "start": pd.Timestamp("2020-11-02"),
            "end": end,
            "multiplier": 1.0,
        },
        "unchanged": {
            "start": pd.Timestamp("2020-11-02"),
            "end": pd.Timestamp("2020-11-30"),
            "multiplier": 1.0,
        },
    }
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


def test_split_policies():
    policies = {
        "bla": {
            "start": "2020-01-01",
            "end": "2020-12-31",
        }
    }

    result_first, result_second = split_policies(
        policies, "2020-02-28", "2020-05-01", "2020-06-01"
    )

    expected_first = {
        "bla_first": {
            "start": pd.Timestamp("2020-02-28"),
            "end": pd.Timestamp("2020-04-30"),
        }
    }

    expected_second = {
        "bla_second": {
            "start": pd.Timestamp("2020-05-01"),
            "end": pd.Timestamp("2020-06-01"),
        }
    }

    assert result_first == expected_first
    assert result_second == expected_second


@pytest.fixture
def policies2():
    d = {
        "always": {},
        "educ_school_0": {},
        "educ_preschool_0": {},
        "educ_nursery_0": {},
        "work_non_recurrent": {},
        "other_recurrent_daily": {},
    }
    return d


def test_remove_work_policies(policies2):
    res = remove_work_policies(policies2)
    expected = {
        "always": {},
        "educ_school_0": {},
        "educ_preschool_0": {},
        "educ_nursery_0": {},
        "other_recurrent_daily": {},
    }
    assert res == expected


def test_remove_educ_policies(policies2):
    res = remove_educ_policies(policies2)
    expected = {
        "always": {},
        "work_non_recurrent": {},
        "other_recurrent_daily": {},
    }
    assert res == expected


def test_remove_other_policies(policies2):
    res = remove_other_policies(policies2)
    expected = {
        "always": {},
        "educ_school_0": {},
        "educ_preschool_0": {},
        "educ_nursery_0": {},
        "work_non_recurrent": {},
    }
    assert res == expected


def test_remove_school_policies(policies2):
    res = remove_school_policies(policies2)
    expected = {
        "always": {},
        "educ_preschool_0": {},
        "educ_nursery_0": {},
        "work_non_recurrent": {},
        "other_recurrent_daily": {},
    }
    assert res == expected


def test_remove_young_educ_policies(policies2):
    res = remove_young_educ_policies(policies2)
    expected = {
        "always": {},
        "educ_school_0": {},
        "work_non_recurrent": {},
        "other_recurrent_daily": {},
    }
    assert res == expected
