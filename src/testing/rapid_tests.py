"""Functions for rapid tests."""
import warnings

import numpy as np
import pandas as pd
from sid.shared import boolean_choices
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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        params_slice = params.loc[
            ("rapid_test_demand", "share_workers_receiving_offer")
        ]
    share_of_workers_with_offer = get_piecewise_linear_interpolation_for_one_day(
        date, params_slice
    )
    accept_share_loc = ("rapid_test_demand", "work", "share_accepting_offer")
    share_workers_accepting_offer = params.loc[accept_share_loc]["value"]
    share_workers_to_be_tested = (
        share_of_workers_with_offer * share_workers_accepting_offer
    )
    workers_getting_tested = _give_rapid_tests_to_some_workers(
        states=states,
        contacts=contacts,
        share_workers_to_be_tested=share_workers_to_be_tested,
    )

    # Abstracting from a lot of heterogeneity, we assume that
    # educ workers and school students get tests twice weekly after Easter
    if date > pd.Timestamp("2021-04-01"):
        educ_test_requests = _test_schools_and_educ_workers(states, contacts)
    else:
        educ_test_requests = pd.Series(False, index=states.index)

    requests_rapid_test = workers_getting_tested | educ_test_requests

    return requests_rapid_test


def _test_schools_and_educ_workers(states, contacts):
    """Return which individuals get a rapid test in an education setting.

    As of April 22, rapid tests in nurseries and preschools were not the
    norm and not mandatory.

    - BY: no mention of preschools and nurseries by the Kultusministerium
        (source: https://bit.ly/3aGpspo, rules starting on 2021-04-12).
        Rapid tests are available for children but voluntary
        (source: https://bit.ly/3xs7Lnc, 2021-03-03)
    - BW: no mandatory tests in preschool and nurseries. Some cities starting
        to offer tests (Stuttgart, source: https://bit.ly/3npzLDk, 2021-04-07)
    - NRW: no mention of mandatory or voluntary tests for preschools and nurseries
        as of 2021-04-22 (source: https://bit.ly/3nrEpAX)

    """
    educ_contact_cols = [col for col in contacts if col.startswith("educ_")]
    # educ_contact_cols are all boolean because all educ models are recurrent
    has_educ_contacts = (contacts[educ_contact_cols]).any(axis=1)
    eligible = states["educ_worker"] | (states["occupation"] == "school")
    too_long_since_last_test = states["cd_received_rapid_test"] <= -3
    to_test = eligible & too_long_since_last_test & has_educ_contacts
    return to_test


def _give_rapid_tests_to_some_workers(states, contacts, share_workers_to_be_tested):
    date = get_date(states)
    work_cols = [col for col in contacts if col.startswith("work_")]
    has_work_contacts = (contacts[work_cols] > 0).any(axis=1)

    # starting 2021-04-26 every worker must be offered two tests per week
    # source: https://bit.ly/2Qw4Md6
    if date > pd.Timestamp("2021-04-26"):
        too_long_since_last_test = states["cd_received_rapid_test"] <= -3
    else:
        # Assume that workers are only tested weekly.
        too_long_since_last_test = states["cd_received_rapid_test"] <= -7

    should_get_test = has_work_contacts & too_long_since_last_test
    truth_probabilities = np.full(len(states), share_workers_to_be_tested)
    receives_offer_and_accepts = pd.Series(
        boolean_choices(truth_probabilities), index=states.index
    )
    to_test = should_get_test & receives_offer_and_accepts
    return to_test
