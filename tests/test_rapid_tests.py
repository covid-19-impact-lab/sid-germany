import numpy as np
import pandas as pd
import pytest

from src.testing.rapid_tests import _give_rapid_tests_to_some_workers
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
    contacts["educ_nursery_0"] = [False, True, False, False, False, False]
    contacts["educ_school_0"] = [True, False, False, False, True, True]
    return contacts


def test_test_schools_and_educ_workers(educ_states, contacts):
    res = _test_schools_and_educ_workers(states=educ_states, contacts=contacts)
    expected = pd.Series(
        [False, True, False, False, False, True], index=educ_states.index
    )
    pd.testing.assert_series_equal(res, expected, check_names=False)


@pytest.fixture
def work_states():
    states = pd.DataFrame()
    # 1: no, because recently tested
    # 2: no, because no contacts today
    # 3: yes, because non recurrent contact
    # 4: yes, because recurrent contact
    # 5: no before the 26th of April, yes after.
    states["cd_received_rapid_test"] = [-1, -10, -10, -10, -5]
    return states


@pytest.fixture
def work_contacts():
    contacts = pd.DataFrame()
    contacts["work_non_recurrent"] = [
        5,  # 1: no, because recently tested
        0,  # 2: no, because no contacts today
        2,  # 3: yes, because non recurrent contact
        0,  # 4: yes, because recurrent contact
        3,  # 5: no before the 26th of April, yes after.
    ]
    contacts["work_recurrent"] = [
        True,  # 1: no, because recently tested
        False,  # 2: no, because no contacts today
        False,  # 3: yes, because non recurrent contact
        True,  # 4: yes, because recurrent contact
        True,  # 5: no before the 26th of April, yes after.
    ]
    # these must not count
    contacts["other_contacts"] = True
    return contacts


def test_give_rapid_tests_to_some_workers_early(work_states, work_contacts):
    work_states["date"] = pd.Timestamp("2021-04-01")
    expected = pd.Series([False, False, True, True, False])
    res = _give_rapid_tests_to_some_workers(work_states, work_contacts, 1.0)
    pd.testing.assert_series_equal(res, expected)


def test_give_rapid_tests_to_some_workers_late(work_states, work_contacts):
    work_states["date"] = pd.Timestamp("2021-05-05")
    expected = pd.Series([False, False, True, True, True])
    res = _give_rapid_tests_to_some_workers(work_states, work_contacts, 1.0)
    pd.testing.assert_series_equal(res, expected)


def test_give_rapid_tests_to_some_workers_no_compliance(work_states, work_contacts):
    work_states["date"] = pd.Timestamp("2021-05-05")
    expected = pd.Series([False, False, False, False, False])
    res = _give_rapid_tests_to_some_workers(work_states, work_contacts, 0.0)
    pd.testing.assert_series_equal(res, expected)


def test_give_rapid_tests_to_some_workers_imperfect_compliance(
    work_states, work_contacts
):
    work_states["date"] = pd.Timestamp("2021-05-05")
    # the result depends on this seed!
    np.random.seed(999)
    expected = pd.Series([False, False, True, False, True])
    res = _give_rapid_tests_to_some_workers(work_states, work_contacts, 0.5)
    pd.testing.assert_series_equal(res, expected)
