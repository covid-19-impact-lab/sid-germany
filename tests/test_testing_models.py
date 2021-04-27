import pandas as pd
import pytest

from src.testing.testing_models import (
    _calculate_positive_tests_to_distribute_per_age_group,
)
from src.testing.testing_models import _demand_test_for_educ_workers
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
    params.index.names = ["category", "subcategory", "name"]
    return params


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
    pd.testing.assert_series_equal(res, expected, check_names=False, check_dtype=False)


def test_demand_test_zero_remainder(states, params):
    positivity_rate_overall = 0.25
    test_shares_by_age_group = pd.Series(
        [0.5, 0.25, 0.25], index=["0-4", "5-14", "15-34"]
    )
    positivity_rate_by_age_group = pd.Series(
        [0.125, 0.25, 0.25], index=["0-4", "5-14", "15-34"]
    )
    params.loc["share_known_cases"] = 1.0

    res = demand_test(
        states=states,
        params=params,
        positivity_rate_overall=positivity_rate_overall,
        test_shares_by_age_group=test_shares_by_age_group,
        positivity_rate_by_age_group=positivity_rate_by_age_group,
        seed=5999,
    )
    expected = states["cd_symptoms_true"] == -1
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_demand_test_zero_remainder_only_half_of_symptomatic_request(states, params):
    params.loc[("test_demand", "symptoms", "share_symptomatic_requesting_test")] = 0.5
    params.loc["share_known_cases"] = 1
    positivity_rate_overall = 0.25
    test_shares_by_age_group = pd.Series(
        [0.5, 0.25, 0.25], index=["0-4", "5-14", "15-34"]
    )
    positivity_rate_by_age_group = pd.Series(
        [0.125, 0.25, 0.25], index=["0-4", "5-14", "15-34"]
    )
    states["infectious"] = states.index.isin([1, 4, 7])
    states["currently_infected"] = states.eval(
        "(infectious | symptomatic | (cd_infectious_true >= 0))"
    )

    expected = pd.Series(False, index=states.index)
    # 1 test for each age group
    # [0, 2, 9] developed symptoms yesterday and are without test
    # 1 is drawn to demand a test because of symptoms: 2
    expected.loc[2] = True
    # after: 1 positive test remaining for 0-4 and 1 remaining for 15-34.
    # in 0-4 only loc=1 is infectious -> that person gets the test
    expected.loc[1] = True
    # in 15-34 only loc=7 is infectious -> that person gets the test
    expected.loc[7] = True

    res = demand_test(
        states=states,
        params=params,
        positivity_rate_overall=positivity_rate_overall,
        test_shares_by_age_group=test_shares_by_age_group,
        positivity_rate_by_age_group=positivity_rate_by_age_group,
        seed=394,
    )
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_demand_test_non_zero_remainder(states, params):
    states["newly_infected"] = True
    states["infectious"] = (
        [True, True] + [True, True, False, False] + [True, False, True, True]
    )
    states["currently_infected"] = states.eval(
        "(infectious | symptomatic | (cd_infectious_true >= 0))"
    )

    # tests to distribute: 2 per individual.
    # 0-4 get one extra. 5-14 are even. 15-34 have two tests removed.
    states["cd_symptoms_true"] = [-1, 2] + [-1, -1, -10, 30] + [-1] * 4

    params.loc["share_known_cases"] = 1
    positivity_rate_overall = 1 / 3
    test_shares_by_age_group = pd.Series([1 / 3] * 3, index=["0-4", "5-14", "15-34"])
    positivity_rate_by_age_group = pd.Series([0.2] * 3, index=["0-4", "5-14", "15-34"])
    res = demand_test(
        states=states,
        params=params,
        positivity_rate_overall=positivity_rate_overall,
        test_shares_by_age_group=test_shares_by_age_group,
        positivity_rate_by_age_group=positivity_rate_by_age_group,
        seed=333,
    )
    # the order of the last four is random and will change if the seed is changed!
    expected = pd.Series(
        [True, True] + [True, True, False, False] + [True, False, False, True],
        index=states.index,
    )
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_demand_test_with_teachers(states, params):
    states["newly_infected"] = True
    states["infectious"] = (
        [True, True] + [True, True, False, False] + [True, False, True, True]
    )
    states["currently_infected"] = states.eval(
        "(infectious | symptomatic | (cd_infectious_true >= 0))"
    )

    # tests to distribute: 2 per individual.
    # 0-4 get one extra. 5-14 are even. 15-34 2 get tests because teacher
    states["cd_symptoms_true"] = [-1, 2] + [-1, -1, -10, 30] + [2, 2, 2, 2]
    states.loc[-2:, "educ_worker"] = True
    states["date"] = pd.Timestamp("2021-03-07")

    params.loc["share_known_cases"] = 1
    positivity_rate_overall = 1 / 3
    test_shares_by_age_group = pd.Series([1 / 3] * 3, index=["0-4", "5-14", "15-34"])
    positivity_rate_by_age_group = pd.Series([0.2] * 3, index=["0-4", "5-14", "15-34"])
    res = demand_test(
        states=states,
        params=params,
        positivity_rate_overall=positivity_rate_overall,
        test_shares_by_age_group=test_shares_by_age_group,
        positivity_rate_by_age_group=positivity_rate_by_age_group,
        seed=333,
    )
    expected = pd.Series(
        [True, True] + [True, True, False, False] + [False, False, True, True],
        index=states.index,
    )
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


def test_demand_test_for_educ_workers(states, params):
    # adjust fixture
    states["educ_worker"] = [False] + [True] * 8 + [False]
    states["state"] = ["Bavaria"] * 6 + ["Hessen"] * 4
    states["infectious"] = [True] + [False] + [True] * 8
    states["currently_infected"] = states.eval(
        "(infectious | symptomatic | (cd_infectious_true >= 0))"
    )

    demanded = pd.Series(False, index=states.index)
    demanded.loc[3] = True
    demanded.loc[9] = True

    states["date"] = pd.Timestamp("2021-03-09")  # Tuesday

    expected = pd.Series(
        [
            # Monday
            False,  # not teacher
            # Tuesday
            False,  # not infectious
            # Wednesday
            False,  # wrong day
            True,  # already demanding test
            # Thursday
            False,  # wrong day
            # Friday
            False,  # wrong day
            False,  # vacation state
            # Saturday
            False,  # vacation state
            # Sunday
            False,  # vacation state
            True,  # already demanding test
        ],
        index=states.index,
    )

    res = _demand_test_for_educ_workers(demanded, states, params)
    pd.testing.assert_series_equal(res, expected, check_names=False)
