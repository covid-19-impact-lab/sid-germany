"""Functions for rapid tests."""
import warnings

import numpy as np
import pandas as pd
from sid.time import get_date

from src.testing.create_rapid_test_statistics import create_rapid_test_statistics
from src.testing.shared import get_piecewise_linear_interpolation_for_one_day


def rapid_test_demand(
    receives_rapid_test,  # noqa: U100
    states,
    params,
    contacts,
    seed,
    save_path=None,
    randomize=False,
    share_refuser=None,
):
    """Assign rapid tests to group.

    Starting after Easter, all education workers and pupils attending school receive a
    test if they participate in school and haven't received a rapid test within 4 days.

    Workers also get tested and more so as time increases.

    Lastly, household members of individuals with symptoms, a positive PCR test
    or a positive rapid test demand a rapid test with 85% probability.

    If randomize is True the calculated demand is distributed randomly in the entire
    population (excluding a share of refusers).

    After Easter vaccinated individuals do not perform rapid tests.

    """
    date = get_date(states)

    # get params subsets
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        work_offer_params = params.loc[
            ("rapid_test_demand", "share_workers_receiving_offer")
        ]
        work_accept_params = params.loc[
            ("rapid_test_demand", "share_accepting_work_offer")
        ]
        educ_workers_params = params.loc[("rapid_test_demand", "educ_worker_shares")]
        students_params = params.loc[("rapid_test_demand", "student_shares")]
        private_demand_params = params.loc[("rapid_test_demand", "private_demand")]

    # get work demand inputs
    share_of_workers_with_offer = get_piecewise_linear_interpolation_for_one_day(
        date, work_offer_params
    )
    share_workers_accepting_offer = get_piecewise_linear_interpolation_for_one_day(
        date, work_accept_params
    )
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
    if date < pd.Timestamp("2021-04-06"):
        freq_tup = ("rapid_test_demand", "educ_frequency", "before_easter")
    else:
        freq_tup = ("rapid_test_demand", "educ_frequency", "after_easter")
    educ_frequency = params.loc[freq_tup, "value"]

    # get household member inputs
    private_demand_share = get_piecewise_linear_interpolation_for_one_day(
        date, private_demand_params
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
        frequency=educ_frequency,
    )

    hh_demand = _calculate_hh_member_rapid_test_demand(
        states=states, demand_share=private_demand_share
    )

    sym_without_pcr_demand = _calculate_own_symptom_rapid_test_demand(
        states=states, demand_share=private_demand_share
    )

    other_contact_demand = _calculate_other_meeting_rapid_test_demand(
        states=states, contacts=contacts, demand_share=private_demand_share
    )

    private_demand = hh_demand | sym_without_pcr_demand | other_contact_demand
    rapid_test_demand = work_demand | educ_demand | private_demand
    preemptive_rapid_test_demand = work_demand | educ_demand | other_contact_demand

    # vaccinated individuals do not test themselves for work, educ or leisure contacts
    if date > pd.Timestamp("2021-04-05"):
        rapid_test_demand = _only_not_fully_vaccinated_test_themselves(
            preemptive_rapid_test_demand, states
        )

    rapid_test_demand = rapid_test_demand | hh_demand | sym_without_pcr_demand

    if randomize and date > pd.Timestamp("2021-04-05"):  # only randomize after Easter
        assert (
            share_refuser is not None
        ), "You must specify a share of individuals that refuse to take a rapid test"

        target_share_to_be_tested = rapid_test_demand.mean()
        rapid_test_demand = _randomize_rapid_tests(
            states=states,
            target_share_to_be_tested=target_share_to_be_tested,
            share_refuser=share_refuser,
            seed=seed,
        )

    if save_path is not None:
        demand_by_channel = pd.DataFrame(
            {
                "private": private_demand,
                "work": work_demand,
                "educ": educ_demand,
                # could also include "hh", "sym_without_pcr", "other_contact"
            }
        )
        if randomize:
            demand_by_channel["random"] = rapid_test_demand

        shares = create_rapid_test_statistics(
            demand_by_channel=demand_by_channel, states=states, date=date, params=params
        )

        if not save_path.exists():  # want to save with columns
            to_add = shares.T.to_csv()
        else:  # want to save without columns
            to_add = shares.T.to_csv().split("\n", 1)[1]
        with open(save_path, "a") as f:
            f.write(to_add)

    return rapid_test_demand


def _calculate_educ_rapid_test_demand(
    states, contacts, educ_worker_multiplier, student_multiplier, frequency
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
        frequency (int): test every [frequency] days

    """
    eligible = _get_eligible_educ_participants(states, contacts, frequency)
    educ_worker_demand = _get_educ_worker_demand(
        eligible, states, educ_worker_multiplier
    )
    student_demand = _get_student_demand(eligible, states, student_multiplier)
    educ_rapid_test_demand = educ_worker_demand | student_demand
    return educ_rapid_test_demand


def _get_eligible_educ_participants(states, contacts, frequency):
    educ_contact_cols = [col for col in contacts if col.startswith("educ_")]
    # educ_contact_cols are all boolean because all educ models are recurrent
    has_educ_contacts = (contacts[educ_contact_cols]).any(axis=1)

    too_long_since_last_test = states["cd_received_rapid_test"] <= -frequency

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


def _calculate_hh_member_rapid_test_demand(states, demand_share):
    """Calculate demand by household members of positive tested and fresh symptomatics.

    Args:
        states (pandas.DataFrame): sid states DataFrame
        demand_share (float): share of household members that request
            a rapid test in response to an event in their household. Individuals
            with a quarantine compliance above 1 - demand_share request
            a rapid test.

    """
    had_event_in_hh = _determine_if_hh_had_event(states)
    would_request_test = states["quarantine_compliance"] >= (1 - demand_share)
    not_tested_within_3_days = states["cd_received_rapid_test"] < -3
    hh_demand = had_event_in_hh & would_request_test & not_tested_within_3_days
    return hh_demand


def _calculate_own_symptom_rapid_test_demand(states, demand_share):
    """Calculate the demand by symptomatic individuals who have no PCR test scheduled.

    We assume that there is no difference in the propensity to take a rapid test
    irrespective of whether it's own symptoms or symptoms in a household member.

    """
    complier = states["quarantine_compliance"] >= (1 - demand_share)
    without_pcr_test = states["cd_received_test_result_true"] < -4
    fresh_symptomatic = states["cd_symptoms_true"].between(-2, 0)
    no_rapid_test_since_symptoms = (
        states["cd_received_rapid_test"] < states["cd_symptoms_true"]
    )
    own_symptom_demand = (
        complier & without_pcr_test & fresh_symptomatic & no_rapid_test_since_symptoms
    )
    return own_symptom_demand


def _calculate_other_meeting_rapid_test_demand(states, contacts, demand_share):
    scaling_factor = 1.0
    demand_share = scaling_factor * demand_share

    complier = states["quarantine_compliance"] >= (1 - demand_share)
    not_tested_recently = states["cd_received_rapid_test"] < -3

    weekly_other_cols = [
        col for col in contacts if col.startswith("other_recurrent_weekly_")
    ]
    with_relevant_contact = (contacts[weekly_other_cols] > 0).any(axis=1)

    to_be_tested = complier & not_tested_recently & with_relevant_contact
    return to_be_tested


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


def _randomize_rapid_tests(states, target_share_to_be_tested, share_refuser, seed):
    np.random.seed(seed)
    # upscale the rapid_test_share to reach the target despite refusers
    willing_to_be_tested = states[states["rapid_test_compliance"] >= share_refuser]
    test_share_among_compliers = target_share_to_be_tested / (1 - share_refuser)
    to_be_tested = np.random.choice(
        a=[True, False],
        size=len(willing_to_be_tested),
        p=[
            test_share_among_compliers,
            1 - test_share_among_compliers,
        ],
    )
    to_test_indices = willing_to_be_tested[to_be_tested].index
    rapid_test_demand = pd.Series(False, index=states.index)
    rapid_test_demand[to_test_indices] = True
    return rapid_test_demand


def _only_not_fully_vaccinated_test_themselves(rapid_test_demand, states):
    """Exclude fully vaccinated individuals from being tested.

    The immunity countdown is initialized at -9999. The vaccine countdown is set
    between -1 and 21. This is only the 1st vaccine. Assuming 30 days between
    shots and 14 days of wait period after the 2nd shot, individuals are freed
    from testing obligations ~45 days after their first shot. This translates
    into countdown values between -45 and -20. For simplicity we simply assume
    that individuals stop testing themselves 40 days after their first shot.

    """
    more_than_14d_since_vaccination = states["cd_is_immune_by_vaccine"].between(
        -9990, -40
    )
    lowered_test_demand = rapid_test_demand & ~more_than_14d_since_vaccination
    return lowered_test_demand
