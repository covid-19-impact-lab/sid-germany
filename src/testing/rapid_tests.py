"""Functions for rapid tests."""
import warnings

import pandas as pd
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

    rapid_test_demand = work_demand | educ_demand

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


def rapid_test_reactions(states, contacts, params, seed):  # noqa: U100
    """Make people react to a positive rapid test.

    source: The COSMO Study of 2021-03-09
    url: https://bit.ly/3gHlcKd (3.5 Verhalten nach positivem Selbsttest)

    - 85% isolate ("isoliere mich und beschränke meine Kontakte bis zur Klärung")
        => We use this multiplier of 0.15 here. We assume households are only
        reduced by 30%, i.e. have a multiplier of 0.7.

    - 85% seek PCR test.
        => This is used in task_build_full_params.

    - 80% inform contacts of last two weeks.
        => This is not supported yet as we do not have contact tracing yet.

    This function is called before `post_process_contacts`.

    """
    contacts = contacts.copy(deep=True)

    received_rapid_test = states["cd_received_rapid_test"] == 0
    pos_rapid_test = states["is_tested_positive_by_rapid_test"]
    quarantine_pool = received_rapid_test & pos_rapid_test

    for col in contacts:
        multiplier = 0.7 if col == "households" else 0.15

        refuser = states["quarantine_compliance"] <= multiplier
        not_staying_home = refuser | ~quarantine_pool
        # no need to worry about dtypes because post_process_contacts happens
        # after this function is called.
        contacts[col] = contacts[col].where(cond=not_staying_home, other=0)

    return contacts
