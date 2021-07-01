import numpy as np
import pandas as pd
import pytest

from src.testing.rapid_tests import _calculate_educ_rapid_test_demand
from src.testing.rapid_tests import _calculate_other_meeting_rapid_test_demand
from src.testing.rapid_tests import _calculate_own_symptom_rapid_test_demand
from src.testing.rapid_tests import _calculate_true_positive_and_false_negatives
from src.testing.rapid_tests import _calculate_weights
from src.testing.rapid_tests import _calculate_work_rapid_test_demand
from src.testing.rapid_tests import _create_rapid_test_statistics
from src.testing.rapid_tests import _determine_if_hh_had_event
from src.testing.rapid_tests import _get_eligible_educ_participants
from src.testing.rapid_tests import _share_demanded_by_infected


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
        frequency=3,
    )
    expected = pd.Series(
        [False, True, False, False, False, True], index=educ_states.index
    )
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_get_eligible_educ_participants_early(educ_states, contacts):
    educ_states["date"] = pd.Timestamp("2021-01-01")
    res = _get_eligible_educ_participants(educ_states, contacts, frequency=7)
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


def test_calculate_own_symptom_rapid_test_demand():
    states = pd.DataFrame(
        columns=[
            "cd_symptoms_true",
            "quarantine_compliance",
            "cd_received_test_result_true",
            "cd_received_rapid_test",
        ],
        data=[
            [-10, 0.9, -99, -99],  # not symptomatic
            [-2, 0.05, -99, -99],  # refuser
            [-1, 0.9, 2, -99],  # pending PCR test
            [-2, 0.9, -99, -1],  # tested since symptoms
            [0, 0.9, -99, -5],  # demands test
        ],
    )
    expected = pd.Series([False, False, False, False, True])
    res = _calculate_own_symptom_rapid_test_demand(states=states, demand_share=0.5)
    pd.testing.assert_series_equal(res, expected)


def test_calculate_other_meeting_rapid_test_demand():
    states = pd.DataFrame()
    states["quarantine_compliance"] = [0.2, 0.8, 0.8, 0.8]
    states["cd_received_rapid_test"] = [-99, -2, -99, -99]

    contacts = pd.DataFrame()
    contacts["other_recurrent_weekly_1"] = [3, 3, 0, 3]
    contacts["other_non_recurrent"] = 2
    demand_share = 0.3

    res = _calculate_other_meeting_rapid_test_demand(
        states=states, contacts=contacts, demand_share=demand_share
    )

    # 0: non-complier, 1: recently tested, 2: no relevant contact, 3: test
    expected = pd.Series([False, False, False, True])
    pd.testing.assert_series_equal(res, expected)


def test_share_demanded_by_infected():
    # 1st: not infected. does not demand test -> should be ignored
    # 2nd: not infected and demands a test -> enters denominator
    # 3rd: infected and does not demand a test -> should be ignored
    # 4th and 5th: infected and demand a test -> enter numerator and denominator
    states = pd.DataFrame({"currently_infected": [False, False, True, True, True]})
    weights = pd.DataFrame({"channel": [0, 0.5, 0.2, 1, 0.5]})
    demand_by_channel = pd.DataFrame({"channel": [False, True, False, True, True]})
    res = _share_demanded_by_infected(
        demand_by_channel=demand_by_channel,
        states=states,
        weights=weights,
        channel="channel",
    )
    expected = 1.5 / (0.5 + 1.5)
    assert res == expected


def test_calculate_weights():
    demand_by_channel = pd.DataFrame(
        {
            "a": [False, False, True, True],
            "b": [False, True, False, True],
        }
    )
    expected = pd.DataFrame(
        {
            "a": [0, 0, 1, 0.5],
            "b": [0, 1, 0, 0.5],
        }
    )
    res = _calculate_weights(demand_by_channel)
    assert expected.equals(res)


def test_create_rapid_test_statistics(monkeypatch):
    date = pd.Timestamp("2021-04-26")
    demand_by_channel = pd.DataFrame(
        {
            "a": [False, False, True, True, False, False, True, True],
            "b": [False, True, False, True, False, True, False, True],
        }
    )
    states = pd.DataFrame(
        {
            "currently_infected": [False, False, True, True, False, False, False, True],
        }
    )

    def mocked_sample_test_outcome(states, receives_rapid_test, params, seed):
        out = pd.Series([True, False] * int(len(states) / 2), index=states.index)
        out[~receives_rapid_test] = False
        return out

    monkeypatch.setattr(
        "src.testing.rapid_tests._sample_test_outcome", mocked_sample_test_outcome
    )

    res = _create_rapid_test_statistics(
        demand_by_channel=demand_by_channel,
        states=states,
        date=date,
        params=None,
    )

    # weights:
    # a: 0, 0, 1, 0.5, 0, 0, 1, 0.5
    # b: 0, 1, 0, 0.5, 0, 1, 0, 0.5
    #
    # groups:
    # a: 2, 3, 6, 7
    # b: 1, 3, 5, 7
    #
    # infected: 2, 3, 7
    #
    # test results overall
    # not tested: 0, 4
    # tested negative: 1, 3, 5, 7
    # tested positive: 2, 6
    #
    # true positive: 2
    # true negative: 1, 5
    # false negative: 3, 7
    # false positive: 6

    expected = pd.DataFrame(
        {
            0: {
                "date": date,
                "n_individuals": 8,
                "share_with_rapid_test_through_a": 3 / 8,
                "share_of_a_rapid_tests_demanded_by_infected": 2 / 3,
                "share_with_rapid_test_through_b": 3 / 8,
                "share_of_b_rapid_tests_demanded_by_infected": 1 / 3,
                "share_with_rapid_test_for_any_reason": 0.75,
                "n_rapid_tests_overall": 6,
                "n_rapid_tests_through_a": 3,
                "n_rapid_tests_through_b": 3,
                # overall shares
                "share_of_rapid_tests_that_are_true_positive": 0.5,
                "share_of_rapid_tests_that_are_true_negative": 0.5,
                "share_of_rapid_tests_that_are_false_negative": 0.5,
                "share_of_rapid_tests_that_are_false_positive": 0.5,
                # shares in a
                "share_of_a_rapid_tests_that_are_true_positive": 0.5,
                "share_of_a_rapid_tests_that_are_true_negative": 0.0,
                "share_of_a_rapid_tests_that_are_false_negative": 1.0,
                "share_of_a_rapid_tests_that_are_false_positive": 0.5,
                # shares in b
                "share_of_b_rapid_tests_that_are_true_positive": np.nan,
                "share_of_b_rapid_tests_that_are_true_negative": 0.5,
                "share_of_b_rapid_tests_that_are_false_negative": 0.5,
                "share_of_b_rapid_tests_that_are_false_positive": np.nan,
            }
        }
    )
    assert set(expected.index) == set(res.index)
    pd.testing.assert_frame_equal(expected.loc[res.index], res)


def test_calculate_true_positive_and_false_negatives():
    # 1: True positive
    # 2: True negative
    # 3: False positive
    # 4 and 5: False negative
    # 6: not tested
    states = pd.DataFrame(
        {
            "currently_infected": [True, False, False, True, True, True],
        }
    )
    rapid_test_results = pd.Series([True, False, True, False, False, False])
    receives_rapid_test = pd.Series([True, True, True, True, True, False])

    (
        res_share_true_positive,
        res_share_false_negative,
    ) = _calculate_true_positive_and_false_negatives(
        states=states,
        rapid_test_results=rapid_test_results,
        receives_rapid_test=receives_rapid_test,
    )

    assert res_share_true_positive == 1 / 2  # 1 and 3 tested positive, 1 is infected
    assert res_share_false_negative == 2 / 3  # 2,4,5 tested negative, 4, 5 infected
