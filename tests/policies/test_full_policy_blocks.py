import pytest

from src.policies.full_policy_blocks import get_lockdown_with_multipliers


def fake_func():
    pass


@pytest.fixture
def contact_models():
    contact_models = {
        "work": {"model": fake_func, "is_recurrent": False},
        "educ_school_0": {"model": fake_func, "is_recurrent": True, "assort_by": "id1"},
        "educ_preschool_0": {
            "model": fake_func,
            "is_recurrent": True,
            "assort_by": "id2",
        },
        "other": {"model": fake_func, "is_recurrent": False},
    }
    return contact_models


@pytest.fixture
def block_info():
    return {"prefix": "prefix", "start_date": "2020-10-23", "end_date": "2020-10-31"}


@pytest.fixture
def start_and_end_multipliers():
    start = {"work": 0.5, "educ": 0.6, "other": 0.7}
    end = {"work": 0.8, "educ": 0.9, "other": 1.0}
    return start, end


@pytest.fixture
def multipliers():
    return {
        "work": {"attend_multiplier": 0.5, "hygiene_multiplier": 0.5},
        "educ": 0.6,
        "other": 0.7,
    }


@pytest.fixture
def expected_keys():
    keys = {
        "prefix_work",
        "prefix_educ_school_0",
        "prefix_educ_preschool_0",
        "prefix_other",
    }
    return keys


def test_get_lockdown_with_multipliers_runs(
    contact_models, block_info, multipliers, expected_keys
):
    res = get_lockdown_with_multipliers(
        contact_models, block_info, multipliers, educ_options={}
    )
    assert res.keys() == expected_keys


def test_get_lockdown_with_multipliers_a_b_schooling(
    contact_models, block_info, multipliers, expected_keys
):
    res = get_lockdown_with_multipliers(
        contact_models,
        block_info,
        multipliers,
        educ_options={
            "school": {
                "a_b_query": None,
                "non_a_b_attend": False,
                "hygiene_multiplier": 0.8,
            }
        },
    )
    assert res.keys() == expected_keys
