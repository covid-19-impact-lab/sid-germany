"""This module contains functions to create unit intervals for rapid tests."""
import numpy as np


def create_unit_intervals_for_rapid_tests(states):
    """Create unit intervals for different subgroups.

    The function loops over the name of the unit interval and the query to select this
    subset.

    """
    for interval_name, query in [
        ("rapid_test_interval_all", "all"),
    ]:
        states = create_unit_interval_for_group(states, interval_name, query, 0)

    return states


def create_unit_interval_for_group(states, interval_name, query, seed):
    """Create unit interval for group.

    First, this function selects the relevant subset with a query and, then, creates a
    unit interval for these people.

    """
    np.random.seed(seed)

    locs = states.index if query == "all" else states.query(query).index

    # shuffle is inplace and can only be used on numpy objects
    locs = np.array(locs)
    np.random.shuffle(locs)

    states.loc[locs, interval_name] = (np.arange(len(locs)) / len(locs)).astype(
        np.float32
    )

    return states
