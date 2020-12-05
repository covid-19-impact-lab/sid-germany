import pytest

from src.policies.full_policy_blocks import get_german_reopening_phase
from src.policies.full_policy_blocks import get_hard_lockdown
from src.policies.full_policy_blocks import get_hard_lockdown_with_ab_schooling
from src.policies.full_policy_blocks import get_only_educ_closed
from src.policies.full_policy_blocks import get_soft_lockdown
from src.policies.full_policy_blocks import get_soft_lockdown_with_ab_schooling


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


def test_get_only_educ_closed_runs(contact_models, block_info):
    get_only_educ_closed(contact_models, block_info)


def test_get_hard_lockdown_runs(contact_models, block_info):
    get_hard_lockdown(contact_models, block_info, 0.5)


def test_get_german_reopening_phase_runs(
    contact_models, block_info, start_and_end_multipliers
):
    get_german_reopening_phase(contact_models, block_info, *start_and_end_multipliers)


def test_get_soft_lockdown_runs(contact_models, block_info, multipliers):
    get_soft_lockdown(contact_models, block_info, multipliers)


def test_get_soft_lockdown_with_ab_schooling(contact_models, block_info, multipliers):
    get_soft_lockdown_with_ab_schooling(contact_models, block_info, multipliers, 12)


def test_get_hard_lockdown_with_ab_schooling(contact_models, block_info, multipliers):
    get_hard_lockdown_with_ab_schooling(contact_models, block_info, multipliers, 12)
