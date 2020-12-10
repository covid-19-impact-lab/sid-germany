from functools import partial

import pytest

from src.policies.domain_level_policy_blocks import _get_base_policy
from src.policies.domain_level_policy_blocks import (
    implement_ab_schooling_with_reduced_other_educ_models,
)
from src.policies.domain_level_policy_blocks import reduce_educ_models
from src.policies.domain_level_policy_blocks import reduce_other_models
from src.policies.domain_level_policy_blocks import reduce_work_models
from src.policies.domain_level_policy_blocks import reopen_educ_models
from src.policies.domain_level_policy_blocks import reopen_other_models
from src.policies.domain_level_policy_blocks import reopen_work_models
from src.policies.domain_level_policy_blocks import shut_down_educ_models
from src.policies.domain_level_policy_blocks import shut_down_other_models
from src.policies.domain_level_policy_blocks import shut_down_work_models
from src.policies.single_policy_functions import implement_a_b_school_system_above_age
from src.policies.single_policy_functions import reduce_recurrent_model
from src.policies.single_policy_functions import reduce_work_model
from src.policies.single_policy_functions import reopen_educ_model_germany
from src.policies.single_policy_functions import reopen_other_model
from src.policies.single_policy_functions import reopen_work_model
from src.policies.single_policy_functions import shut_down_model
from src.policies.single_policy_functions import shut_down_work_model


def compare_policy_dicts(d1, d2):
    """Needed because partialed functions are not considered the same."""
    assert d1.keys() == d2.keys()
    for key, d1_dict in d1.items():
        d2_dict = d2[key]
        assert d1_dict.keys() == d2_dict.keys(), f"Not same keys in {key}."
        for inner_key, val1 in d1_dict.items():
            val2 = d2_dict[inner_key]
            if isinstance(val1, partial):
                assert val1.func == val2.func
                assert val1.args == val2.args
                assert val1.keywords == val2.keywords
            else:
                assert val1 == val2


def fake_func():
    pass


@pytest.fixture
def contact_models():
    contact_models = {
        "work1": {"model": fake_func, "is_recurrent": False},
        "work2": {"model": fake_func, "is_recurrent": True},
        "educ1": {"model": fake_func, "is_recurrent": True},
        "other1": {"model": fake_func, "is_recurrent": False},
    }
    return contact_models


def test_shut_down_work_models(contact_models):
    res = shut_down_work_models(
        contact_models,
        {
            "start_date": "2020-10-01",
            "end_date": "2020-10-31",
            "prefix": "test_shut_down",
        },
    )
    expected = {
        "test_shut_down_work1": {
            "affected_contact_model": "work1",
            "start": "2020-10-01",
            "end": "2020-10-31",
            "policy": shut_down_work_model,
        },
        "test_shut_down_work2": {
            "affected_contact_model": "work2",
            "start": "2020-10-01",
            "end": "2020-10-31",
            "policy": shut_down_work_model,
        },
    }
    compare_policy_dicts(res, expected)


def test_shut_down_educ_models(contact_models):
    res = shut_down_educ_models(
        contact_models,
        {
            "start_date": "2020-10-01",
            "end_date": "2020-10-31",
            "prefix": "sd_educ",
        },
    )

    expected = {
        "sd_educ_educ1": {
            "affected_contact_model": "educ1",
            "start": "2020-10-01",
            "end": "2020-10-31",
            "policy": shut_down_model,
        },
    }
    assert res == expected


def test_shut_down_other_models(contact_models):
    res = shut_down_other_models(
        contact_models,
        {
            "start_date": "2020-10-01",
            "end_date": "2020-10-31",
            "prefix": "test_shut_down",
        },
    )
    expected = {
        "test_shut_down_other1": {
            "affected_contact_model": "other1",
            "start": "2020-10-01",
            "end": "2020-10-31",
            "policy": shut_down_model,
        },
    }
    compare_policy_dicts(res, expected)


def test_reduce_work_models(contact_models):
    block_info = {
        "start_date": "2020-10-10",
        "end_date": "2020-10-20",
        "prefix": "reduce_work",
    }
    res = reduce_work_models(contact_models, block_info, 0.5)
    expected = {
        "reduce_work_work1": {
            "affected_contact_model": "work1",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": partial(reduce_work_model, multiplier=0.5),
        },
        "reduce_work_work2": {
            "affected_contact_model": "work2",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": partial(reduce_work_model, multiplier=0.5),
        },
    }
    compare_policy_dicts(res, expected)


def test_reduce_educ_models(contact_models):
    block_info = {
        "start_date": "2020-10-10",
        "end_date": "2020-10-20",
        "prefix": "reduce",
    }
    res = reduce_educ_models(contact_models, block_info, 0.5)
    expected = {
        "reduce_educ1": {
            "affected_contact_model": "educ1",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": partial(reduce_recurrent_model, multiplier=0.5),
        }
    }
    compare_policy_dicts(res, expected)


def test_reduce_other_models(contact_models):
    contact_models["other2"] = {"model": fake_func, "is_recurrent": True}
    block_info = {
        "start_date": "2020-10-10",
        "end_date": "2020-10-20",
        "prefix": "reduce",
    }
    res = reduce_other_models(contact_models, block_info, 0.5)
    expected = {
        "reduce_other1": {
            "affected_contact_model": "other1",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": 0.5,
        },
        "reduce_other2": {
            "affected_contact_model": "other2",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": partial(reduce_recurrent_model, multiplier=0.5),
        },
    }
    compare_policy_dicts(res, expected)


def test_reopen_educ_models(contact_models):
    block_info = {
        "start_date": "2020-06-06",
        "end_date": "2020-06-20",
        "prefix": "reopen",
    }

    res = reopen_educ_models(
        contact_models=contact_models,
        block_info=block_info,
        start_multiplier=0.2,
        end_multiplier=0.8,
        switching_date="2020-06-15",
        reopening_dates=None,
    )
    expected = {
        "reopen_educ1": {
            "start": "2020-06-06",
            "end": "2020-06-20",
            "affected_contact_model": "educ1",
            "policy": partial(
                reopen_educ_model_germany,
                start_multiplier=0.2,
                end_multiplier=0.8,
                switching_date="2020-06-15",
                reopening_dates=None,
                is_recurrent=True,
            ),
        }
    }
    compare_policy_dicts(res, expected)


def test_reopen_work_models(contact_models):
    block_info = {
        "start_date": "2020-10-10",
        "end_date": "2020-10-20",
        "prefix": "reduce",
    }

    res = reopen_work_models(
        contact_models, block_info, start_multiplier=0.2, end_multiplier=0.8
    )
    work_func = partial(
        reopen_work_model,
        start_multiplier=0.2,
        end_multiplier=0.8,
        start_date=block_info["start_date"],
        end_date=block_info["end_date"],
    )
    expected = {
        "reduce_work1": {
            "affected_contact_model": "work1",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": work_func,
        },
        "reduce_work2": {
            "affected_contact_model": "work2",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": work_func,
        },
    }
    compare_policy_dicts(res, expected)


def test_reopen_other_models(contact_models):
    contact_models["other2"] = {"model": fake_func, "is_recurrent": True}
    block_info = {
        "start_date": "2020-10-10",
        "end_date": "2020-10-20",
        "prefix": "reopen",
    }
    res = reopen_other_models(
        contact_models, block_info, start_multiplier=0.2, end_multiplier=0.8
    )

    expected = {
        "reopen_other1": {
            "start": "2020-10-10",
            "end": "2020-10-20",
            "affected_contact_model": "other1",
            "policy": partial(
                reopen_other_model,
                start_multiplier=0.2,
                end_multiplier=0.8,
                start_date="2020-10-10",
                end_date="2020-10-20",
                is_recurrent=False,
            ),
        },
        "reopen_other2": {
            "start": "2020-10-10",
            "end": "2020-10-20",
            "affected_contact_model": "other2",
            "policy": partial(
                reopen_other_model,
                start_multiplier=0.2,
                end_multiplier=0.8,
                start_date="2020-10-10",
                end_date="2020-10-20",
                is_recurrent=True,
            ),
        },
    }
    compare_policy_dicts(res, expected)


def test_implement_ab_schooling_with_reduced_other_educ_models():
    block_info = {
        "start_date": "2020-10-10",
        "end_date": "2020-10-20",
        "prefix": "test",
    }

    contact_models = {
        "educ_school_1": {
            "model": fake_func,
            "is_recurrent": True,
            "assort_by": ["school_id_1"],
        },
        "educ_school_2": {
            "model": fake_func,
            "is_recurrent": True,
            "assort_by": ["school_id_2"],
        },
        "educ_preschool": {
            "model": fake_func,
            "is_recurrent": True,
            "assort_by": ["preschool_id"],
        },
        "educ_nursery": {
            "model": fake_func,
            "is_recurrent": True,
            "assort_by": ["nursery_id"],
        },
        "other": {},
    }
    res = implement_ab_schooling_with_reduced_other_educ_models(
        contact_models, block_info, age_cutoff=10, multiplier=0.5
    )
    expected = {
        "test_educ_school_1": {
            "affected_contact_model": "educ_school_1",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": partial(
                implement_a_b_school_system_above_age,
                age_cutoff=10,
            ),
        },
        "test_educ_school_2": {
            "affected_contact_model": "educ_school_2",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": partial(
                implement_a_b_school_system_above_age,
                age_cutoff=10,
            ),
        },
        "test_educ_preschool": {
            "affected_contact_model": "educ_preschool",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": partial(reduce_recurrent_model, multiplier=0.5),
        },
        "test_educ_nursery": {
            "affected_contact_model": "educ_nursery",
            "start": "2020-10-10",
            "end": "2020-10-20",
            "policy": partial(reduce_recurrent_model, multiplier=0.5),
        },
    }
    compare_policy_dicts(res, expected)


def test_get_base_policy():
    res = _get_base_policy(
        "model_name", {"start_date": "2020-10-01", "end_date": "2020-10-10"}
    )
    expected = {
        "affected_contact_model": "model_name",
        "start": "2020-10-01",
        "end": "2020-10-10",
    }
    assert res == expected