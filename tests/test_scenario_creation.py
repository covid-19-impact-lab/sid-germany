from functools import partial

import pandas as pd
import pytest

from src.config import VERY_LATE
from src.contact_models.get_contact_models import get_household_contact_model
from src.contact_models.get_contact_models import get_other_non_recurrent_contact_model
from src.contact_models.get_contact_models import get_work_non_recurrent_contact_model
from src.policies.enacted_policies import HYGIENE_MULTIPLIER
from src.policies.policy_tools import combine_dictionaries
from src.policies.single_policy_functions import reduce_work_model
from src.simulation.params_scenarios import _build_new_date_params
from src.simulation.params_scenarios import (
    _change_piecewise_linear_parameter_to_fixed_value_after_date,
)
from src.simulation.scenario_simulation_inputs import (
    _get_policies_with_different_work_attend_multiplier_after_date,
)


@pytest.fixture
def params():
    params = pd.DataFrame()
    params["value"] = [0.3, 0.6, 0.9]
    params["name"] = ["2021-01-01", "2021-04-01", "2021-04-30"]
    params["subcategory"] = "subcategory"
    params["category"] = "category"
    params = params.set_index(["category", "subcategory", "name"])
    params.loc[("other", "other", "other")] = 15
    return params


def test_build_new_date_params(params):
    res = _build_new_date_params(
        params.loc[("category", "subcategory")],
        change_date=pd.Timestamp("2021-04-17"),
        new_val=1.0,
    )
    expected_index = pd.DatetimeIndex(
        ["2021-01-01", "2021-04-01", "2021-04-16", "2021-04-17", "2025-12-31"],
        name="name",
    )
    expected = pd.DataFrame(index=expected_index)
    expected["value"] = [
        0.3,  # kept
        0.6,  # kept
        0.7551724137931035,  # interpolated value right before the change
        1.0,  # on the change date
        1.0,  # maintain value
    ]

    pd.testing.assert_frame_equal(res, expected)


def test_change_piecewise_linear_parameter_to_fixed_value_after_date(params):
    res = _change_piecewise_linear_parameter_to_fixed_value_after_date(
        params=params,
        loc=("category", "subcategory"),
        change_date="2021-05-15",
        new_val=0.3,
    )

    expected = params.copy(deep=True)
    expected.loc[("category", "subcategory", "2021-05-14")] = 0.9
    expected.loc[("category", "subcategory", "2021-05-15")] = 0.3
    expected.loc[("category", "subcategory", "2025-12-31")] = 0.3

    pd.testing.assert_frame_equal(res, expected, check_like=True)


def test_get_policies_with_different_work_attend_multiplier_after_date():
    contact_models = combine_dictionaries(
        [
            get_household_contact_model(),
            get_work_non_recurrent_contact_model(),
            get_other_non_recurrent_contact_model(),
        ]
    )

    enacted_policies = {
        "keep_work": {
            "affected_contact_model": "work_non_recurrent",
            "start": pd.Timestamp("2021-01-01"),
            "end": pd.Timestamp("2021-02-28"),
            "policy": 0.5,
        },
        "cut_work": {
            "affected_contact_model": "work_non_recurrent",
            "start": pd.Timestamp("2021-03-01"),
            "end": pd.Timestamp("2021-04-30"),
            "policy": 0.5,
        },
        "drop_work": {
            "affected_contact_model": "work_non_recurrent",
            "start": pd.Timestamp("2021-05-01"),
            "end": pd.Timestamp("2021-05-31"),
            "policy": 0.5,
        },
        "other": {
            "affected_contact_model": "other_non_recurrent",
            "start": pd.Timestamp("2021-01-01"),
            "end": pd.Timestamp("2021-05-31"),
            "policy": 0.5,
        },
    }
    res = _get_policies_with_different_work_attend_multiplier_after_date(
        enacted_policies=enacted_policies,
        contact_models=contact_models,
        new_attend_multiplier=0.0,
        split_date=pd.Timestamp("2021-04-15"),
        prefix="test",
    )
    expected_work_policy = partial(
        reduce_work_model,
        attend_multiplier=0.0,
        hygiene_multiplier=HYGIENE_MULTIPLIER,
        is_recurrent=False,
    )
    expected = {
        "keep_work": {
            "affected_contact_model": "work_non_recurrent",
            "start": pd.Timestamp("2021-01-01"),
            "end": pd.Timestamp("2021-02-28"),
            "policy": 0.5,
        },
        "cut_work_first": {
            "affected_contact_model": "work_non_recurrent",
            "start": pd.Timestamp("2021-03-01"),
            "end": pd.Timestamp("2021-04-14"),
            "policy": 0.5,
        },
        "other_first": {
            "affected_contact_model": "other_non_recurrent",
            "start": pd.Timestamp("2021-01-01"),
            "end": pd.Timestamp("2021-04-14"),
            "policy": 0.5,
        },
        "other_second": {
            "affected_contact_model": "other_non_recurrent",
            "start": pd.Timestamp("2021-04-15"),
            "end": pd.Timestamp("2021-05-31"),
            "policy": 0.5,
        },
        "test_work_non_recurrent": {
            "affected_contact_model": "work_non_recurrent",
            "start": pd.Timestamp("2021-04-15"),
            "end": VERY_LATE,
            "policy": expected_work_policy,
        },
    }

    # This is a custom comparison because the two partialed functions are not recognized
    # to be identical
    assert res.keys() == expected.keys()
    for pol_name, exp_pol in expected.items():
        res_pol = res[pol_name]
        assert res_pol.keys() == exp_pol.keys()
        if pol_name == "test_work_non_recurrent":
            for key, val in exp_pol.items():
                if key == "policy":
                    assert res_pol["policy"].func == val.func
                    assert res_pol["policy"].args == val.args
                    assert res_pol["policy"].keywords == val.keywords
                else:
                    assert res_pol[key] == val
        else:
            assert exp_pol == res_pol
