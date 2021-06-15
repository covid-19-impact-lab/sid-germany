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
