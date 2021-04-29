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
    work_compliance_multiplier = (
        share_of_workers_with_offer * share_workers_accepting_offer
    )
    work_demand = _calculate_work_rapid_test_demand(
        states=states,
        contacts=contacts,
        compliance_multiplier=work_compliance_multiplier,
    )

    # Abstracting from a lot of heterogeneity, we assume that
    # educ workers and school students get tests twice weekly after Easter
    if date > pd.Timestamp("2021-04-01"):
        educ_demand = _calculate_educ_rapid_test_demand(states, contacts)
    else:
        educ_demand = pd.Series(False, index=states.index)

    rapid_test_demand = work_demand | educ_demand

    return rapid_test_demand


def _calculate_educ_rapid_test_demand(states, contacts):
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
    to_test = should_get_test & receives_offer_and_accepts
    return to_test
