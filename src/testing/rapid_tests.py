"""Functions for rapid tests."""
import pandas as pd
from sid.time import get_date


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

    # Abstracting from a lot of heterogeneity, we assume that
    # educ workers and school students get tests twice weekly after Easter
    if date > pd.Timestamp("2021-04-01"):
        educ_test_requests = _test_schools_and_educ_workers(states, contacts)
    else:
        educ_test_requests = pd.Series(False, index=states.index)

    requests_rapid_test = educ_test_requests

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
