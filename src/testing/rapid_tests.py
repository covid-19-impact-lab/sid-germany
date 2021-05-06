"""Functions for rapid tests."""
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from sid.time import get_date

from src.testing.shared import get_piecewise_linear_interpolation_for_one_day


def rapid_test_demand(
    receives_rapid_test,  # noqa: U100
    states,
    params,  # noqa: U100
    contacts,
    seed,  # noqa: U100
):
    """Assign rapid tests to group.

    Starting after Easter, all education workers and pupils attending school receive a
    test if they participate in school and haven't received a rapid test within 4 days.

    Workers also get tested and more so as time increases.

    Lastly, household members of individuals with symptoms, a positive PCR test
    or a positive rapid test demand a rapid test with 85% probability.

    """
    date = get_date(states)

    # get params subsets
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        params_slice = params.loc[
            ("rapid_test_demand", "share_workers_receiving_offer")
        ]
        educ_workers_params = params.loc[("rapid_test_demand", "educ_worker_shares")]
        students_params = params.loc[("rapid_test_demand", "student_shares")]
        hh_member_params = params.loc[("rapid_test_demand", "hh_member_demand")]

    # get work demand inputs
    share_of_workers_with_offer = get_piecewise_linear_interpolation_for_one_day(
        date, params_slice
    )
    accept_share_loc = ("rapid_test_demand", "work", "share_accepting_offer")
    share_workers_accepting_offer = params.loc[accept_share_loc]["value"]
    work_compliance_multiplier = (
        share_of_workers_with_offer * share_workers_accepting_offer
    )

    # get educ demand inputs
    educ_worker_multiplier = get_piecewise_linear_interpolation_for_one_day(
        date, educ_workers_params
    )
    student_multiplier = get_piecewise_linear_interpolation_for_one_day(
        date, students_params
    )

    # get housheold member inputs
    hh_member_demand_share = get_piecewise_linear_interpolation_for_one_day(
        date, hh_member_params
    )

    work_demand = _calculate_work_rapid_test_demand(
        states=states,
        contacts=contacts,
        compliance_multiplier=work_compliance_multiplier,
    )

    educ_demand = _calculate_educ_rapid_test_demand(
        states=states,
        contacts=contacts,
        educ_worker_multiplier=educ_worker_multiplier,
        student_multiplier=student_multiplier,
    )

    hh_demand = _calculate_hh_member_rapid_test_demand(
        states=states, hh_member_demand_share=hh_member_demand_share
    )

    rapid_test_demand = work_demand | educ_demand | hh_demand

    return rapid_test_demand


def _calculate_educ_rapid_test_demand(
    states, contacts, educ_worker_multiplier, student_multiplier
):
    """Return which individuals get a rapid test in an education setting.

    Args:
        states (pandas.DataFrame): states DataFrame
        contacts (pandas.DataFrame): DataFrame with the same index as states.
            columns are the contact model names. All education contact models start
            with `educ_`. All education columns are recurrent, i.e. are boolean.
        educ_worker_multiplier (float): share of educ workers that have not been
            tested long enough and have education contacts that receive and
            accept a test.
        student_multiplier (float): share of school students that have not been
            tested long enough and have education contacts that receive and
            accept a test.

    """
    eligible = _get_eligible_educ_participants(states, contacts)
    educ_worker_demand = _get_educ_worker_demand(
        eligible, states, educ_worker_multiplier
    )
    student_demand = _get_student_demand(eligible, states, student_multiplier)
    educ_rapid_test_demand = educ_worker_demand | student_demand
    return educ_rapid_test_demand


def _get_eligible_educ_participants(states, contacts):
    date = get_date(states)
    educ_contact_cols = [col for col in contacts if col.startswith("educ_")]
    # educ_contact_cols are all boolean because all educ models are recurrent
    has_educ_contacts = (contacts[educ_contact_cols]).any(axis=1)

    # Assume weekly tests before Easter and twice weekly tests after Easter
    # We should get a fade-in through different ends of Easter vaccation
    if date < pd.Timestamp("2021-04-06"):
        too_long_since_last_test = states["cd_received_rapid_test"] <= -7
    else:
        too_long_since_last_test = states["cd_received_rapid_test"] <= -3

    eligible = has_educ_contacts & too_long_since_last_test
    return eligible


def _get_educ_worker_demand(eligible, states, educ_worker_multiplier):
    eligible_educ_workers = eligible & states["educ_worker"]
    educ_worker_cutoff = 1 - educ_worker_multiplier
    educ_worker_demand = eligible_educ_workers & (
        states["rapid_test_compliance"] > educ_worker_cutoff
    )
    return educ_worker_demand


def _get_student_demand(eligible, states, student_multiplier):
    eligible_students = eligible & (states["occupation"] == "school")
    student_cutoff = 1 - student_multiplier
    student_demand = eligible_students & (
        states["rapid_test_compliance"] > student_cutoff
    )
    return student_demand


def _calculate_work_rapid_test_demand(states, contacts, compliance_multiplier):
    date = get_date(states)
    work_cols = [col for col in contacts if col.startswith("work_")]
    has_work_contacts = (contacts[work_cols] > 0).any(axis=1)

    # starting 2021-04-26 every worker must be offered two tests per week
    # source: https://bit.ly/2Qw4Md6
    # To have a gradual transition we gradually increase the test frequency
    if date < pd.Timestamp("2021-04-07"):  # before Easter
        allowed_days_btw_tests = 7
    elif date < pd.Timestamp("2021-04-13"):
        allowed_days_btw_tests = 6
    elif date < pd.Timestamp("2021-04-20"):
        allowed_days_btw_tests = 5
    elif date < pd.Timestamp("2021-04-27"):
        allowed_days_btw_tests = 4
    else:  # date > pd.Timestamp("2021-04-26")
        allowed_days_btw_tests = 3

    too_long_since_last_test = (
        states["cd_received_rapid_test"] <= -allowed_days_btw_tests
    )

    should_get_test = has_work_contacts & too_long_since_last_test
    complier = states["rapid_test_compliance"] >= (1 - compliance_multiplier)
    receives_offer_and_accepts = should_get_test & complier
    work_rapid_test_demand = should_get_test & receives_offer_and_accepts
    return work_rapid_test_demand


def _calculate_hh_member_rapid_test_demand(states, hh_member_demand_share):
    """Calculate demand by household members of positive tested and fresh symptomatics.

    Args:
        states (pandas.DataFrame): sid states DataFrame
        hh_member_demand_share (float): share of household members that request
            a rapid test in response to an event in their household. Individuals
            with a quarantine compliance above 1 - hh_member_demand_share request
            a rapid test.

    """
    had_event_in_hh = _determine_if_hh_had_event(states)
    would_request_test = states["quarantine_compliance"] >= (1 - hh_member_demand_share)
    not_tested_within_3_days = states["cd_received_rapid_test"] < -3
    hh_demand = had_event_in_hh & would_request_test & not_tested_within_3_days
    return hh_demand


def _determine_if_hh_had_event(states):
    """Determine who had a potential rapid test triggering event in their household.

    Returns:
        had_event_in_hh (pandas.Series): Series with the same index as states.
            True for individuals where a household member got symptoms yesterday,
            who received a positive rapid test yesterday or who have a new known
            case in their household.

    """
    rapid_test_event = (states["cd_received_rapid_test"] == -1) & (
        states["is_tested_positive_by_rapid_test"]
    )
    is_event = (
        rapid_test_event | states["new_known_case"] | (states["cd_symptoms_true"] == -1)
    )
    had_event_in_hh = is_event.groupby(states["hh_id"]).transform(np.any)

    return had_event_in_hh


def rapid_test_reactions(states, contacts, params, seed):  # noqa: U100
    """Make people react to a positive rapid tests"""
    contacts = contacts.copy(deep=True)

    # we assume that if you haven't received PCR confirmation within 7 days
    # you go back to having contacts.
    received_rapid_test = states["cd_received_rapid_test"].between(
        -5, 0, inclusive=True
    )
    pos_rapid_test = states["is_tested_positive_by_rapid_test"]
    quarantine_pool = received_rapid_test & pos_rapid_test

    for col in contacts:
        loc = ("rapid_test_demand", "reaction")
        if col == "households":
            multiplier = params.loc[(*loc, "hh_contacts_multiplier"), "value"]
        else:
            multiplier = params.loc[(*loc, "not_hh_contacts_multiplier"), "value"]
        refuser = states["quarantine_compliance"] <= multiplier
        not_staying_home = refuser | ~quarantine_pool
        # no need to worry about dtypes because post_process_contacts happens
        # after this function is called.

        if is_bool_dtype(contacts[col]):
            contacts[col] = contacts[col].where(cond=not_staying_home, other=False)
        else:
            contacts[col] = contacts[col].where(cond=not_staying_home, other=0)

    return contacts
