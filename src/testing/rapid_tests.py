"""Functions for rapid tests."""
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

    Starting after Easter, all education workers and pupils attending school receive a
    test if they participate in school and haven't received a rapid test within 4 days.

    """
    date = get_date(states)

    # Abstracting from a lot of heterogeneity, we assume that
    # educ workers and school students get tests twice weekly after Easter
    if date > pd.Timestamp("2021-04-01"):
        educ_test_requests = _test_schools_and_educ_workers(states, contacts)
    else:
        educ_test_requests = pd.Series(False, index=states.index)

    requests_rapid_test = educ_test_requests

    return requests_rapid_test


def _test_schools_and_educ_workers(states, contacts):
    eligible = states[
        "educ_worker"
    ]  # | (states["occupation"] == "school")  # for verification
    untested_within_4_days = states["cd_received_rapid_test"] <= -4
    educ_contact_cols = [col for col in contacts if col.startswith("educ_")]
    has_educ_contacts = (contacts[educ_contact_cols] > 1).any(axis=1)
    to_test = eligible & untested_within_4_days & has_educ_contacts
    return to_test
