import numpy as np
import pandas as pd
import pytest

from src.testing.rapid_tests import _calculate_educ_rapid_test_demand
from src.testing.rapid_tests import _calculate_work_rapid_test_demand
from src.testing.rapid_tests import _get_eligible_educ_participants
from src.testing.rapid_tests import rapid_test_reactions


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
    states["cd_received_rapid_test"] = [-2, -5, -20, -5, -20, -5]
    states["rapid_test_compliance"] = 1.0
    states["date"] = pd.Timestamp("2021-05-05")
    return states


@pytest.fixture
def contacts(educ_states):
    contacts = pd.DataFrame(index=educ_states.index)
    contacts["households"] = 2
    contacts["educ_nursery_0"] = [False, True, False, False, True, False]
    contacts["educ_school_0"] = [True, False, False, False, True, True]
    return contacts


def test_calculate_educ_rapid_test_demand(educ_states, contacts):
    res = _calculate_educ_rapid_test_demand(
        states=educ_states,
        contacts=contacts,
        educ_worker_multiplier=1,
        student_multiplier=1,
    )
    expected = pd.Series(
        [False, True, False, False, False, True], index=educ_states.index
    )
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_get_eligible_educ_participants_early(educ_states, contacts):
    educ_states["date"] = pd.Timestamp("2021-01-01")
    res = _get_eligible_educ_participants(educ_states, contacts)
    expected = pd.Series(
        [
            False,  # recently tested
            False,  # recently enough for early, should be True after Easter
            False,  # no contacts (wrong occupation)
            False,  # no contacts (student)
            True,  # nursery kid that has contacts and has not been recently tested
            False,  # student recently for early, should be True after Easter
        ],
    )
    pd.testing.assert_series_equal(res, expected)


@pytest.fixture
def work_states():
    states = pd.DataFrame()
    # 1: no, because recently tested
    # 2: no, because no contacts today
    # 3: yes, because non recurrent contact
    # 4: yes, because recurrent contact
    # 5: no before the 26th of April, yes after.
    # 6: yes if compliance_multiplier >= 0.3
    states["cd_received_rapid_test"] = [-1, -10, -10, -10, -5, -10]
    states["rapid_test_compliance"] = [0.8, 0.8, 0.8, 0.8, 0.8, 0.3]
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
        6,  # 6: yes if compliance multiplier >= 0.7
    ]
    contacts["work_recurrent"] = [
        True,  # 1: no, because recently tested
        False,  # 2: no, because no contacts today
        False,  # 3: yes, because non recurrent contact
        True,  # 4: yes, because recurrent contact
        True,  # 5: no before the 26th of April, yes after.
        False,  # 6: yes if compliance_multiplier >= 0.7
    ]
    # these must not count
    contacts["other_contacts"] = True
    contacts["other_contacts_2"] = 20
    return contacts


def test_calculate_work_rapid_test_demand_early(work_states, work_contacts):
    work_states["date"] = pd.Timestamp("2021-04-01")
    expected = pd.Series([False, False, True, True, False, True])
    res = _calculate_work_rapid_test_demand(
        work_states, work_contacts, compliance_multiplier=1.0  # perfect compliance
    )
    pd.testing.assert_series_equal(res, expected)


def test_calculate_work_rapid_test_demand_late(work_states, work_contacts):
    work_states["date"] = pd.Timestamp("2021-05-05")
    expected = pd.Series([False, False, True, True, True, False])
    res = _calculate_work_rapid_test_demand(
        work_states, work_contacts, compliance_multiplier=0.3
    )
    pd.testing.assert_series_equal(res, expected)


def test_calculate_work_rapid_test_demand_no_compliance(work_states, work_contacts):
    work_states["date"] = pd.Timestamp("2021-05-05")
    expected = pd.Series([False, False, False, False, False, False])
    res = _calculate_work_rapid_test_demand(
        work_states, work_contacts, compliance_multiplier=0.0
    )
    pd.testing.assert_series_equal(res, expected)


def test_calculate_work_rapid_test_demand_imperfect_compliance(
    work_states, work_contacts
):
    work_states["date"] = pd.Timestamp("2021-05-05")
    expected = pd.Series([False, False, True, True, True, False])
    res = _calculate_work_rapid_test_demand(
        work_states, work_contacts, compliance_multiplier=0.5
    )
    pd.testing.assert_series_equal(res, expected)


def test_rapid_test_reactions():
    states = pd.DataFrame()
    states["quarantine_compliance"] = [0.0, 0.2, 0.4, 0.6, 0.8]

    contacts = pd.DataFrame()
    contacts["households"] = [True, True, True, True, True]
    contacts["other_recurrent"] = [False, True, False, True, False]
    contacts["other_non_recurrent"] = [5, 2, 2, 2, 2]

    expected = pd.DataFrame()
    expected["households"] = [True, True, True, True, 0]
    expected["other_recurrent"] = [False, False, False, False, False]
    expected["other_non_recurrent"] = [5, 0, 0, 0, 0]

    res = rapid_test_reactions(states, contacts, None, None)

    pd.testing.assert_frame_equal(res, expected, check_dtype=False)


def test_rapid_test_reactions_lln():
    states = pd.DataFrame()
    states["quarantine_compliance"] = np.random.uniform(0, 1, size=10000)

    contacts = pd.DataFrame()
    contacts["households"] = [True] * 10000
    contacts["other"] = True

    res = rapid_test_reactions(states, contacts, None, None)

    share_meet_hh = res["households"].mean()
    share_meet_other = res["other"].mean()
    assert 0.65 < share_meet_hh < 0.75
    assert 0.10 < share_meet_other < 0.20
