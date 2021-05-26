from pandas.api.types import is_bool_dtype


def rapid_test_reactions(states, contacts, params, seed):  # noqa: U100
    """Make people react to a positive rapid tests by reducing their contacts."""
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
