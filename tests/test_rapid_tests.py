import pandas as pd
import pytest

from src.testing.rapid_tests import _test_schools_and_educ_workers


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
