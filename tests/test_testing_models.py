import pandas as pd
import pytest

from src.testing.testing_models import (
    _calculate_positive_tests_to_distribute_per_age_group,
)
from src.testing.testing_models import _request_pcr_test_bc_of_symptoms
from src.testing.testing_models import _scale_demand_up_or_down
from src.testing.testing_models import allocate_tests
from src.testing.testing_models import demand_test
from src.testing.testing_models import process_tests

DATE = pd.Timestamp("2020-10-10")


@pytest.fixture(scope="function")
def states():
    states = pd.DataFrame()
    ages = ["0-4"] * 2 + ["5-14"] * 4 + ["15-34"] * 4
    states["age_group_rki"] = pd.Series(ages, dtype="category")
    states["pending_test"] = False
    states["knows_immune"] = False
    states["date"] = DATE
    states["symptomatic"] = False
    states["cd_infectious_true"] = -10
    # 1, 1, 2 infections => 4 newly_infected
    states["newly_infected"] = [True, False, True] + [False] * 5 + [True, True]
    # 0, 2 and 9 are potential symptom test seekers b/c of recent symptoms
    states["cd_symptoms_true"] = [-1, 2, -1] + [-5] * 6 + [-1]
    states["educ_worker"] = False
    states["state"] = "Hessen"
    states["cd_received_test_result_true"] = -3
    states["cd_received_rapid_test"] = -99
    states["index"] = states.index
    return states


@pytest.fixture(scope="function")
def params():
    share_tuple = ("test_demand", "symptoms", "share_symptomatic_requesting_test")
    params = pd.DataFrame(
        1.0,
        columns=["value"],
        index=pd.MultiIndex.from_tuples([share_tuple]),
    )
    params.loc[("FürImmerferien", "Hessen", "start")] = 1601503200  # 2020-10-01
    params.loc[("FürImmerferien", "Hessen", "end")] = 1635631200  # 2021-10-31
    params.loc[
        ("share_known_cases", "share_known_cases", pd.Timestamp("2020-01-01"))
    ] = -1.0
    params.loc[
        ("share_known_cases", "share_known_cases", pd.Timestamp("2022-01-01"))
    ] = -1.0
    rapid_tests_tuple = (
        "test_demand",
        "rapid_tests",
        "share_w_positive_rapid_test_requesting_test",
    )
    params.loc[rapid_tests_tuple] = 0.0

    params.index.names = ["category", "subcategory", "name"]
    return params


# -----------------------------------------------------------------------------


def test_scale_demand_up_or_down(states):
    demanded = pd.Series([True, True] + [False, True] * 4, index=states.index)
    states["infectious"] = True
    states["currently_infected"] = states.eval(
        "(infectious | symptomatic | (cd_infectious_true >= 0))"
    )
    remaining = pd.Series([-2, 0, 2], index=["0-4", "5-14", "15-34"])
    expected_vals = [False, False] + [False, True] * 2 + [True] * 4
    expected = pd.Series(data=expected_vals, index=states.index)
    res = _scale_demand_up_or_down(
        demanded=demanded, states=states, remaining=remaining
    )
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


def test_allocate_tests(states):
    demands_test = pd.Series(True, index=states.index)
    res = allocate_tests(
        n_allocated_tests=None,
        demands_test=demands_test,
        states=None,
        params=None,
        seed=332,
    )
    demands_test[:5] = False
    assert res.all()


def test_process_tests(states):
    expected = pd.Series([True] * 5 + [False] * 5, index=states.index)
    states["pending_test"] = expected.copy(deep=True)
    res = process_tests(
        n_to_be_processed_tests=None, states=states, params=None, seed=13222
    )
    states.loc[:3, "pending_test"] = False
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_demand_test(states, params):
    positivity_rate_overall = 0.25
    test_shares_by_age_group = pd.Series(
        [0.5, 0.25, 0.25], index=["0-4", "5-14", "15-34"]
    )
    positivity_rate_by_age_group = pd.Series(
        [0.125, 0.25, 0.25], index=["0-4", "5-14", "15-34"]
    )
    params.loc["share_known_cases"] = 1.0

    assert False, "`demand_test` is not tested at the moment."


# ----------------------------------------------------------------------------


@pytest.fixture
def symptom_states():
    states = pd.DataFrame()
    # 0th and1st didn't get symptoms yesterday
    # 2nd has a pending test, 3rd knows she's immune
    # 4th requests a test
    states["cd_symptoms_true"] = [-5, 3, -1, -1, -1]
    states["pending_test"] = [False, False, True, False, False]
    states["knows_immune"] = [False, False, False, True, False]
    return states


def test_request_pcr_test_bc_of_symptoms_no_one(symptom_states):
    res = _request_pcr_test_bc_of_symptoms(symptom_states, 0.0)
    expected = pd.Series(False, index=symptom_states.index)
    pd.testing.assert_series_equal(res, expected)


def test_reqst_pcr_test_bc_of_symptoms_everyone(symptom_states):
    res = _request_pcr_test_bc_of_symptoms(symptom_states, 1.0)
    expected = pd.Series([False, False, False, False, True], index=symptom_states.index)
    pd.testing.assert_series_equal(res, expected)
