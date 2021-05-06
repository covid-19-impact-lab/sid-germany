import numpy as np
import pandas as pd
import pytest

from src.testing.rapid_tests import _calculate_educ_rapid_test_demand
from src.testing.rapid_tests import _calculate_work_rapid_test_demand
from src.testing.rapid_tests import _determine_if_hh_had_event
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
    states["cd_received_rapid_test"] = 0
    states["is_tested_positive_by_rapid_test"] = True

    contacts = pd.DataFrame()
    contacts["households"] = [True, True, True, True, True]
    contacts["other_recurrent"] = [False, True, False, True, False]
    contacts["other_non_recurrent"] = [5, 2, 2, 2, 2]

    expected = pd.DataFrame()
    expected["households"] = [True, True, True, True, 0]
    expected["other_recurrent"] = [False, False, False, False, False]
    expected["other_non_recurrent"] = [5, 0, 0, 0, 0]

    params = pd.DataFrame(
        data=[0.7, 0.15],
        columns=["value"],
        index=pd.MultiIndex.from_tuples(
            [
                ("rapid_test_demand", "reaction", "hh_contacts_multiplier"),
                ("rapid_test_demand", "reaction", "not_hh_contacts_multiplier"),
            ]
        ),
    )
    res = rapid_test_reactions(states, contacts, params, None)

    pd.testing.assert_frame_equal(res, expected, check_dtype=False)


def test_rapid_test_reactions_lln():
    np.random.seed(38484)
    states = pd.DataFrame()
    states["quarantine_compliance"] = np.random.uniform(0, 1, size=10000)
    states["cd_received_rapid_test"] = [0] * 9900 + [-3] * 90 + [-99] * 10
    states["is_tested_positive_by_rapid_test"] = (
        [True] * 9980 + [False] * 10 + [True] * 10
    )

    contacts = pd.DataFrame()
    contacts["households"] = [True] * 10000
    contacts["other"] = True

    params = pd.DataFrame(
        data=[0.7, 0.15],
        columns=["value"],
        index=pd.MultiIndex.from_tuples(
            [
                ("rapid_test_demand", "reaction", "hh_contacts_multiplier"),
                ("rapid_test_demand", "reaction", "not_hh_contacts_multiplier"),
            ]
        ),
    )

    res = rapid_test_reactions(states, contacts, params, None)

    quarantine_pool = res.loc[:9979]
    share_meet_other = quarantine_pool["other"].mean()
    share_meet_hh = quarantine_pool["households"].mean()
    assert 0.145 < share_meet_other < 0.155
    assert 0.695 < share_meet_hh < 0.705
    assert (res.loc[9980:] == contacts.loc[9980:]).all().all()


def test_determine_if_hh_had_event():
    # these are family members without events
    base_df = pd.DataFrame()
    base_df["cd_received_rapid_test"] = [-5, 5, 0]
    base_df["is_tested_positive_by_rapid_test"] = [True, False, False]
    base_df["cd_symptoms_true"] = [3, -3, -44]
    base_df["new_known_case"] = False

    no_event_hh = base_df.copy(deep=True)
    no_event_hh["hh_id"] = 1
    res = _determine_if_hh_had_event(no_event_hh)
    expected = pd.Series(False, index=no_event_hh.index)
    pd.testing.assert_series_equal(res, expected)

    pos_rapid_test_hh = base_df.copy(deep=True)
    pos_rapid_test_hh["hh_id"] = 2
    # positive rapid test
    pos_rapid_test_hh.loc[3] = {
        "cd_received_rapid_test": -1,
        "is_tested_positive_by_rapid_test": True,
        "cd_symptoms_true": 2,
        "new_known_case": False,
        "hh_id": 2,
    }
    res = _determine_if_hh_had_event(pos_rapid_test_hh)
    expected2 = pd.Series(True, index=pos_rapid_test_hh.index)
    pd.testing.assert_series_equal(res, expected2)

    hh_with_symptom = base_df.copy(deep=True)
    hh_with_symptom["hh_id"] = 3
    hh_with_symptom.loc[3] = {
        "cd_received_rapid_test": -33,
        "is_tested_positive_by_rapid_test": False,
        "cd_symptoms_true": -1,
        "new_known_case": False,
        "hh_id": 3,
    }
    res = _determine_if_hh_had_event(hh_with_symptom)
    expected3 = pd.Series(True, index=hh_with_symptom.index)
    pd.testing.assert_series_equal(res, expected3)

    hh_with_new_known_case = base_df.copy(deep=True)
    hh_with_new_known_case["hh_id"] = 3
    hh_with_new_known_case.loc[3] = {
        "cd_received_rapid_test": -33,
        "is_tested_positive_by_rapid_test": False,
        "cd_symptoms_true": -5,
        "new_known_case": True,
        "hh_id": 3,
    }
    res = _determine_if_hh_had_event(hh_with_new_known_case)
    expected4 = pd.Series(True, index=hh_with_new_known_case.index)
    pd.testing.assert_series_equal(res, expected4)

    full = pd.concat(
        [no_event_hh, pos_rapid_test_hh, hh_with_symptom, hh_with_new_known_case]
    )
    res = _determine_if_hh_had_event(full)
    expected = pd.concat([expected, expected2, expected3, expected4])
    pd.testing.assert_series_equal(res, expected)
