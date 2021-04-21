import numpy as np
import pandas as pd
import pytest

from src.testing.rapid_tests import _request_rapid_test_bc_of_symptoms
from src.testing.rapid_tests import _test_schools_and_educ_workers


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


def test_request_rapid_test_bc_of_symptoms_no_one(symptom_states):
    res = _request_rapid_test_bc_of_symptoms(symptom_states, 0.0)
    expected = np.array([False, False, False, False, False])
    np.testing.assert_equal(res, expected)


def test_request_rapid_test_bc_of_symptoms_everyone(symptom_states):
    res = _request_rapid_test_bc_of_symptoms(symptom_states, 1.0)
    expected = np.array([False, False, False, False, True])
    np.testing.assert_equal(res, expected)


@pytest.fixture
def educ_states():
    states = pd.DataFrame()

    states["educ_worker"] = [True, True, False, False, False, False]
    states["occupation"] = [
        "school_teacher",  # 0: recently tested teacher
        "nursery_teacher",  # 1: teacher that's due to be tested
        "retired",  # 2: retired
        "school",  # 3: student without contacts
        "nursery",  # 4: not school student
        "school",  # 5: student to be tested
    ]
    states["cd_received_rapid_test"] = [-2, -5, -20, -5, -5, -5]
    return states


@pytest.fixture
def contacts(educ_states):
    contacts = pd.DataFrame(index=educ_states.index)
    contacts["households"] = 2
    contacts["educ_nursery_0"] = [0, 4, 0, 0, 0, 0]
    contacts["educ_school_0"] = [10, 0, 0, 0, 4, 10]
    return contacts


def test_test_schools_and_educ_workers(educ_states, contacts):
    res = _test_schools_and_educ_workers(states=educ_states, contacts=contacts)
    expected = pd.Series(
        [False, True, False, False, False, True], index=educ_states.index
    )
    pd.testing.assert_series_equal(res, expected, check_names=False)
