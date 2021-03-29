"""This module contains functions to assign rapid tests."""
from typing import Tuple

import pandas as pd
from sid import get_date


def assign_rapid_test_to_group(
    receives_rapid_test,  # noqa: U100
    states,
    params,  # noqa: U100
    contacts,  # noqa: U100
    seed,  # noqa: U100
    unit_interval_column,
    n_days,
    n_repetitions_per_week,
):
    """Assign rapid tests to group."""
    date = get_date(states)

    intervals = _get_intervals_for_people_receiving_rapid_tests(
        date.weekday(), n_days, n_repetitions_per_week
    )
    query = _build_query_from_intervals(intervals, unit_interval_column)

    loc = states[unit_interval_column].query(query)

    new_receives_rapid_test = pd.Series(index=states.index, data=False)
    new_receives_rapid_test.loc[loc] = True

    return new_receives_rapid_test


def _get_intervals_for_people_receiving_rapid_tests(
    day: int, n_days: int, n_repetitions_per_week: int
) -> Tuple[float]:
    """Get intervals for people receiving rapid tests.

    This function returns a parts of the unit interval which will receive tests.

    Examples
    --------

    If we want to test every single person once a week, we will test people
    with values in [0, 1/7) on the first date, people with values in [1/7, 2/7) on the
    second day, ..., and people with values in [6/7, 1] on the last weekday.

    >>> _get_intervals_for_people_receiving_rapid_tests(0, 7, 1)
    [(0.0, 0.14285714285714285)]
    >>> _get_intervals_for_people_receiving_rapid_tests(6, 7, 1)
    [(0.8571428571428571, 1.001)]

    If we want to test all people twice a week, on Thursday, we retest people who were
    already tested on Monday.

    >>> _get_intervals_for_people_receiving_rapid_tests(3, 7, 2)
    [(0.8571428571428571, 1.001), (0, 0.1428571428571428)]

    """
    interval_length_per_day = n_repetitions_per_week / n_days
    start = day * interval_length_per_day
    end = (day + 1) * interval_length_per_day

    if int(start) != int(end) and end % 1 == 0:
        intervals = [(start % 1, 1.001)]
    elif int(start) != int(end):
        intervals = [(start % 1, 1.001), (0, end % 1)]
    else:
        intervals = [(start % 1, end % 1)]

    return intervals


def _build_query_from_intervals(intervals: Tuple[float], column_name: str) -> str:
    """Build a query from the given intervals."""
    sub_queries = []
    for (start, end) in intervals:
        sub_query = f"(({start} <= {column_name}) & ({column_name} < {end}))"
        sub_queries.append(sub_query)

    return " | ".join(sub_queries)
