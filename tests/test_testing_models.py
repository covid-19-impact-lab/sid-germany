import pandas as pd
import pytest

from src.testing.testing_models import _calculate_test_demand_from_rapid_tests
from src.testing.testing_models import _calculate_test_demand_from_share_known_cases
from src.testing.testing_models import allocate_tests
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
    states["cd_received_rapid_test"] = -99
    states["index"] = states.index
    return states


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


@pytest.fixture
def symptom_states():
    states = pd.DataFrame()
    # 0th and 1st didn't get symptoms yesterday
    # 2nd has a pending test, 3rd knows she's immune
    # 4th requests a test
    states["cd_symptoms_true"] = [-5, 3, -1, -1, -1]
    states["pending_test"] = [False, False, True, False, False]
    states["knows_immune"] = [False, False, False, True, False]
    return states


def test_calculate_test_demand_from_rapid_tests():
    states = pd.DataFrame()
    # cases:
    # 0: not tested today
    # 1: false negative
    # 2: false positive
    # 3: true positive
    # 4: true negative
    states["cd_received_rapid_test"] = [-2, 0, 0, 0, 0]
    states["is_tested_positive_by_rapid_test"] = [True, False, True, True, False]
    states["currently_infected"] = [False, True, False, True, False]

    res = _calculate_test_demand_from_rapid_tests(states, 1)
    expected = pd.Series([False, False, False, True, False], index=states.index)
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_calculate_test_demand_from_share_known_cases():
    states = pd.DataFrame()

    states["newly_infected"] = [True, True, True] + [False] * 7
    states["symptomatic"] = [True, True] + [False] * 8
    states["pending_test"] = [False, True] + [True] * 7 + [False]
    states["currently_infected"] = [True, True, True] + [False] * 5 + [True] * 2
    states["knows_immune"] = False
    states["cd_received_test_result_true"] = -28

    share_known_cases = 2 / 3
    share_of_tests_for_symptomatics = 0.5

    expected = pd.Series([True] + [False] * 8 + [True])

    calculated = _calculate_test_demand_from_share_known_cases(
        states=states,
        share_known_cases=share_known_cases,
        share_of_tests_for_symptomatics=share_of_tests_for_symptomatics,
    )

    pd.testing.assert_series_equal(calculated, expected)
