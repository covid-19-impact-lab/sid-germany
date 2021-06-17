from pathlib import Path

import pandas as pd
import pytest

from src.simulation import params_scenarios
from src.simulation.task_save_params_changes_of_params_scenarios import (
    create_comparison_df,
)
from src.simulation.task_save_params_changes_of_params_scenarios import (
    get_params_scenarios,
)
from src.testing.shared import get_piecewise_linear_interpolation


SCENARIO_FUNCS = [func for name, func in get_params_scenarios()]


@pytest.fixture(scope="function")
def params():
    params = pd.read_pickle(Path(__file__).parent / "params.pkl")
    return params


@pytest.mark.parametrize("func", SCENARIO_FUNCS)
def test_no_side_effect(func, params):
    before = params.copy(deep=True)
    func(params)
    assert before.equals(params)


def test_baseline(params):
    new_params = params_scenarios.baseline(params)
    assert new_params.equals(params)


def test_rapid_tests_at_school_every_day_after_april_5(params):
    new_params = params_scenarios.rapid_tests_at_school_every_day_after_april_5(params)
    comparison = create_comparison_df(new_params=new_params, old_params=params)
    before = params.loc[
        ("rapid_test_demand", "educ_frequency", "after_easter"), "value"
    ]
    expected = pd.DataFrame(
        {
            "category": ["rapid_test_demand"],
            "subcategory": ["educ_frequency"],
            "name": ["after_easter"],
            "before": [before],
            "after": [1.0],
        }
    ).set_index(["category", "subcategory", "name"])
    assert comparison.equals(expected)


def test_rapid_tests_at_school_every_other_day_after_april_5(params):
    new_params = params_scenarios.rapid_tests_at_school_every_other_day_after_april_5(
        params
    )
    comparison = create_comparison_df(new_params=new_params, old_params=params)
    before = params.loc[
        ("rapid_test_demand", "educ_frequency", "after_easter"), "value"
    ]
    expected = pd.DataFrame(
        {
            "category": ["rapid_test_demand"],
            "subcategory": ["educ_frequency"],
            "name": ["after_easter"],
            "before": [before],
            "after": [2.0],
        }
    ).set_index(["category", "subcategory", "name"])
    assert comparison.equals(expected)


def test_no_rapid_tests_at_schools_after_easter(params):
    new_params = params_scenarios.no_rapid_tests_at_schools_after_easter(params)
    comparison = create_comparison_df(new_params=new_params, old_params=params)
    before = params.loc[
        ("rapid_test_demand", "educ_frequency", "after_easter"), "value"
    ]
    expected = pd.DataFrame(
        {
            "category": ["rapid_test_demand"],
            "subcategory": ["educ_frequency"],
            "name": ["after_easter"],
            "before": [before],
            "after": [1000.0],
        }
    ).set_index(["category", "subcategory", "name"])
    assert comparison.equals(expected)


def test_no_seasonality(params):
    new_params = params_scenarios.no_seasonality(params)
    comparison = create_comparison_df(new_params=new_params, old_params=params)
    expected = pd.DataFrame(
        {
            "category": ["seasonality_effect", "seasonality_effect"],
            "subcategory": ["seasonality_effect", "seasonality_effect"],
            "name": ["strong", "weak"],
            "before": [0.3, 0.2],
            "after": [0.0, 0.0],
        }
    ).set_index(["category", "subcategory", "name"])
    assert comparison.equals(expected)


def test_no_private_test_demand(params):
    new_params = params_scenarios.no_private_rapid_test_demand(params)
    comparison = create_comparison_df(new_params=new_params, old_params=params)
    loc = ("rapid_test_demand", "private_demand")
    private_demand_params = params.loc[loc, "value"]
    should_have_changed = private_demand_params[private_demand_params != 0]

    expected = pd.DataFrame()
    expected["before"] = should_have_changed
    expected["name"] = should_have_changed.index
    expected["after"] = 0.0
    expected["category"] = "rapid_test_demand"
    expected["subcategory"] = "private_demand"
    expected = expected.set_index(["category", "subcategory", "name"])
    assert comparison.equals(expected)

    private_series = get_piecewise_linear_interpolation(
        params_slice=new_params.loc[loc, "value"]
    )
    assert (private_series == 0).all()


def test_keep_work_offer_share_at_23_pct_after_easter(params):
    new_params = params_scenarios.keep_work_offer_share_at_23_pct_after_easter(params)
    offer_params = new_params.loc[
        ("rapid_test_demand", "share_workers_receiving_offer"), "value"
    ]
    time_series = get_piecewise_linear_interpolation(params_slice=offer_params)

    assert time_series["2021-04-06"] == 0.23
    assert (time_series["2021-04-06":] == 0.23).all()


def test_mandatory_work_rapid_tests_after_easter(params):
    new_params = params_scenarios.mandatory_work_rapid_tests_after_easter(params)
    accept_params = new_params.loc[
        ("rapid_test_demand", "share_accepting_work_offer"), "value"
    ]
    accept_time_series = get_piecewise_linear_interpolation(params_slice=accept_params)
    assert (accept_time_series[:"2021-04-05"] < 0.95).all()
    assert (accept_time_series["2021-04-06":] == 0.95).all()

    offer_params = new_params.loc[
        ("rapid_test_demand", "share_workers_receiving_offer"), "value"
    ]
    offer_time_series = get_piecewise_linear_interpolation(params_slice=offer_params)
    assert (offer_time_series[:"2021-04-05"] < 0.95).all()
    assert (offer_time_series["2021-04-06":] == 0.95).all()


def test_no_rapid_tests_at_work(params):
    new_params = params_scenarios.no_rapid_tests_at_work(params)
    accept_params = new_params.loc[
        ("rapid_test_demand", "share_accepting_work_offer"), "value"
    ]
    accept_time_series = get_piecewise_linear_interpolation(params_slice=accept_params)
    assert (accept_time_series == 0).all()

    offer_params = new_params.loc[
        ("rapid_test_demand", "share_workers_receiving_offer"), "value"
    ]
    offer_time_series = get_piecewise_linear_interpolation(params_slice=offer_params)
    assert (offer_time_series == 0).all()


def test_no_rapid_tests_at_school(params):
    new_params = params_scenarios.no_rapid_tests_at_schools(params)
    _check_school_demand_is_zero(new_params)


def test_no_rapid_tests_at_work_and_private(params):
    new_params = params_scenarios.no_rapid_tests_at_work_and_private(params)
    _check_no_work_rapid_tests(new_params)
    private_slice = new_params.loc[("rapid_test_demand", "private_demand"), "value"]
    private_series = get_piecewise_linear_interpolation(params_slice=private_slice)
    assert (private_series == 0).all()


def test_no_rapid_tests_at_schools_and_work(params):
    new_params = params_scenarios.no_rapid_tests_at_schools_and_work(params)
    _check_no_work_rapid_tests(new_params)
    _check_school_demand_is_zero(new_params)


def test_no_rapid_tests_at_schools_and_private(params):
    new_params = params_scenarios.no_rapid_tests_at_schools_and_private(params)
    _check_school_demand_is_zero(new_params)

    private_slice = new_params.loc[("rapid_test_demand", "private_demand"), "value"]
    private_series = get_piecewise_linear_interpolation(params_slice=private_slice)
    assert (private_series == 0).all()


def test_start_all_rapid_tests_after_easter(params):
    new_params = params_scenarios.start_all_rapid_tests_after_easter(params)
    slice_tuples = [
        ("rapid_test_demand", "private_demand"),
        ("rapid_test_demand", "educ_worker_shares"),
        ("rapid_test_demand", "student_shares"),
        ("rapid_test_demand", "share_accepting_work_offer"),
        ("rapid_test_demand", "share_workers_receiving_offer"),
    ]

    for loc in slice_tuples:
        params_slice = new_params.loc[loc, "value"]
        time_series = get_piecewise_linear_interpolation(params_slice=params_slice)
        assert (time_series[:"2021-04-05"] == 0).all()
        assert (time_series["2021-04-06":] > 0).all()


# ======================================================================================


def _check_school_demand_is_zero(new_params):
    teacher_params = new_params.loc[
        ("rapid_test_demand", "educ_worker_shares"), "value"
    ]
    teacher_time_series = get_piecewise_linear_interpolation(
        params_slice=teacher_params
    )
    assert (teacher_time_series == 0).all()

    student_params = new_params.loc[("rapid_test_demand", "student_shares"), "value"]
    student_time_series = get_piecewise_linear_interpolation(
        params_slice=student_params
    )
    assert (student_time_series == 0).all()


def _check_no_work_rapid_tests(new_params):
    accept_params = new_params.loc[
        ("rapid_test_demand", "share_accepting_work_offer"), "value"
    ]
    accept_time_series = get_piecewise_linear_interpolation(params_slice=accept_params)
    assert (accept_time_series == 0).all()

    offer_params = new_params.loc[
        ("rapid_test_demand", "share_workers_receiving_offer"), "value"
    ]
    offer_time_series = get_piecewise_linear_interpolation(params_slice=offer_params)
    assert (offer_time_series == 0).all()
