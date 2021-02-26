import pytest

from src.policies.full_policy_blocks import get_german_reopening_phase
from src.policies.full_policy_blocks import get_lockdown_with_multipliers


def fake_func():
    pass


@pytest.fixture
def contact_models():
    contact_models = {
        "work": {"model": fake_func, "is_recurrent": False},
        "educ_school": {"model": fake_func, "is_recurrent": True},
        "educ_preschool": {"model": fake_func, "is_recurrent": True},
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
    return {"work": 0.5, "educ": 0.6, "other": 0.7}


@pytest.fixture
def expected_keys():
    keys = {
        "prefix_work",
        "prefix_educ_school",
        "prefix_educ_preschool",
        "prefix_other",
    }
    return keys


def test_get_german_reopening_phase_runs(
    contact_models, block_info, start_and_end_multipliers, expected_keys
):
    res = get_german_reopening_phase(
        contact_models, block_info, *start_and_end_multipliers
    )
    assert res.keys() == expected_keys


def test_get_lockdown_with_multipliers_runs(
    contact_models, block_info, multipliers, expected_keys
):
    res = get_lockdown_with_multipliers(contact_models, block_info, multipliers)
    assert res.keys() == expected_keys
