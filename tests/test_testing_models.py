import pandas as pd
import pytest

from src.testing.testing_models import (
    _calculate_positive_tests_to_distribute_per_age_group,
)
from src.testing.testing_models import _up_or_downscale_demand
from src.testing.testing_models import demand_test

DATE = pd.Timestamp("2020-10-10")


@pytest.fixture(scope="function")
def states():
    states = pd.DataFrame()
    ages = ["0-4"] * 2 + ["5-14"] * 4 + ["15-34"] * 4
    states["age_group_rki"] = pd.Series(ages, dtype="category")
    states["pending_test"] = False
    states["knows_immune"] = False
    states["date"] = DATE
    # 1, 1, 2 infections => 4 newly_infected
    states["newly_infected"] = [True, False, True] + [False] * 5 + [True, True]
    states["symptomatic"] = [True, False, True] + [False] * 6 + [True]
    return states


def test_up_or_downscale_demand(states):
    states["demanded"] = [True, True] + [False, True] * 4
    remaining = pd.Series([-2, 0, 2], index=["0-4", "5-14", "15-34"])
    expected_vals = [False, False] + [False, True] * 2 + [True] * 4
    expected = pd.Series(data=expected_vals, index=states.index)
    res = _up_or_downscale_demand(states=states, remaining=remaining)
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_calculate_positive_tests_to_distribute_per_age_group():
    n_newly_infected = 20
    share_known_cases = 0.5
    positivity_rate_overall = 0.2
    test_shares_by_age_group = pd.Series(
        [0.4, 0.4, 0.2], index=["0-4", "5-14", "15-34"]
    )
    positivity_rate_by_age_group = pd.Series(
        [0.05, 0.2, 0.5], index=["0-4", "5-14", "15-34"]
    )
    res = _calculate_positive_tests_to_distribute_per_age_group(
        n_newly_infected,
        share_known_cases,
        positivity_rate_overall,
        test_shares_by_age_group,
        positivity_rate_by_age_group,
    )
    expected = pd.Series([1, 4, 5], index=["0-4", "5-14", "15-34"])
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_demand_test_zero_remainder(states):
    params = None
    share_known_cases = 1
    positivity_rate_overall = 0.25
    test_shares_by_age_group = pd.Series(
        [0.5, 0.25, 0.25], index=["0-4", "5-14", "15-34"]
    )
    positivity_rate_by_age_group = pd.Series(
        [0.125, 0.25, 0.25], index=["0-4", "5-14", "15-34"]
    )
    res = demand_test(
        states=states,
        params=params,
        share_known_cases=share_known_cases,
        positivity_rate_overall=positivity_rate_overall,
        test_shares_by_age_group=test_shares_by_age_group,
        positivity_rate_by_age_group=positivity_rate_by_age_group,
    )
    expected = states["symptomatic"].copy(deep=True)
    pd.testing.assert_series_equal(res, expected, check_names=False)
