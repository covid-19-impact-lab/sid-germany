"""Functions for rapid tests."""
import numpy as np
import pandas as pd
from sid.time import get_date


def rapid_test_demand(
    receives_rapid_test,  # noqa: U100
    states,
    params,  # noqa: U100
    contacts,  # noqa: U100
    seed,  # noqa: U100
):
    """Assign rapid tests to group.

    Symptomatic people request a test with the `share_symptomatic_requesting_test`
    probability.

    Starting after Easter, all education workers and pupils attending school receive a
    test if they participate in school and haven't received a rapid test within 4 days.

    """
    date = get_date(states)

    symptom_tuple = ("test_demand", "symptoms", "share_symptomatic_requesting_test")
    share_symptomatic = params.loc[symptom_tuple, "value"]
    if share_symptomatic > 1.0 or share_symptomatic < 0:
        raise ValueError(
            "The share of symptomatic individuals requesting a test must lie in the "
            f"[0, 1] interval, you specified {share_symptomatic}"
        )

    symptomatic_requests = _request_rapid_test_bc_of_symptoms(states, share_symptomatic)

    # Abstracting from a lot of heterogeneity, we assume that
    # educ workers and school students get tests twice weekly after Easter
    if date > pd.Timestamp("2021-04-01"):
        educ_test_requests = _test_schools_and_educ_workers(states, contacts)
    else:
        educ_test_requests = pd.Series(False, index=states.index)

    requests_rapid_test = symptomatic_requests | educ_test_requests

    return requests_rapid_test


def _request_rapid_test_bc_of_symptoms(states, share_symptomatic_requesting_test):
    """Return who requests a rapid test because of symptoms.

    Args:

    Returns:
        requests_rapid_test

    """
    developed_symptoms_yesterday = states["cd_symptoms_true"] == -1
    untested = ~states["pending_test"] & ~states["knows_immune"]
    symptomatic_without_test = developed_symptoms_yesterday & untested
    if share_symptomatic_requesting_test == 1.0:
        requests_rapid_test_locs = states[symptomatic_without_test].index
    else:
        # this ignores the designated number of tests per age group.
        # Adjusting the number of tests to the designated number is done in
        # `_scale_demand_up_or_down` below.
        n_to_demand = int(
            share_symptomatic_requesting_test * symptomatic_without_test.sum()
        )
        pool = states[symptomatic_without_test].index
        requests_rapid_test_locs = np.random.choice(
            size=n_to_demand, a=pool, replace=False
        )

    requests_rapid_test = states.index.isin(requests_rapid_test_locs)
    return requests_rapid_test


def _test_schools_and_educ_workers(states, contacts):
    eligible = states["educ_worker"] | (states["occupation"] == "school")
    untested_within_4_days = states["cd_received_rapid_test"] <= -4
    educ_contact_cols = [col for col in contacts if col.startswith("educ_")]
    has_educ_contacts = (contacts[educ_contact_cols] > 1).any(axis=1)
    to_test = eligible & untested_within_4_days & has_educ_contacts
    return to_test
