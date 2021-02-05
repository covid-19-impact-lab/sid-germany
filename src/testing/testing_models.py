"""Testing models, adjusted from Tobi's sid tutorial."""
import warnings

import numpy as np
import pandas as pd
from sid.time import get_date


def demand_test(
    states,
    params,  # noqa: U100
    share_known_cases,
    positivity_rate_overall,
    test_shares_by_age_group,
    positivity_rate_by_age_group,
):
    """Test demand function.

    This demand model assumes that individuals request a test for a
    Covid-19 infection if they experience symptoms, have not requested
    a test before which is still pending and have not received a positive
    test result with a probability of 1.

    We then add tests among the remaining currently infectious. ###

    Args:
        states (pandas.DataFrame): The states of the individuals.
        params (pandas.DataFrame): A DataFrame with parameters.
        share_known_cases (pandas.Series): share of infections that is detected.
        positivity_rate_overall (pandas.Series): share of total tests that was positive.
        test_shares_by_age_group (pandas.Series or pandas.DataFrame):
            share of tests that was administered to each age group. If a Series the
            index are the age groups. If a DataFrame, the index are the dates and
            the columns are the age groups.
        positivity_rate_by_age_group (pandas.Series or pandas.DataFrame):
            share of tests that was positive in each age group. If a Series the
            index are the age groups. If a DataFrame, the index are the dates and
            the columns are the age groups.

    Returns:
        demand_probability (numpy.ndarray, pandas.Series): An array or a series
            which contains the probability for each individual demanding a test.

    """
    n_newly_infected = states["newly_infected"].sum()
    date = get_date(states)
    if isinstance(test_shares_by_age_group, pd.DataFrame):
        test_shares_by_age_group = test_shares_by_age_group.loc[date]
    if isinstance(positivity_rate_by_age_group, pd.DataFrame):
        positivity_rate_by_age_group = positivity_rate_by_age_group.loc[date]
    if isinstance(positivity_rate_overall, pd.Series):
        positivity_rate_overall = positivity_rate_overall.loc[date]
    if isinstance(share_known_cases, pd.Series):
        share_known_cases = share_known_cases.loc[date]

    n_pos_tests_for_each_group = _calculate_positive_tests_to_distribute_per_age_group(
        n_newly_infected=n_newly_infected,
        share_known_cases=share_known_cases,
        positivity_rate_overall=positivity_rate_overall,
        test_shares_by_age_group=test_shares_by_age_group,
        positivity_rate_by_age_group=positivity_rate_by_age_group,
    )
    states = states.copy()
    states["demanded"] = (
        states["symptomatic"] & ~states["pending_test"] & ~states["knows_immune"]
    )
    demands_by_age_group = states.groupby("age_group_rki")["demanded"].sum()
    remaining = n_pos_tests_for_each_group - demands_by_age_group
    demanded = _up_or_downscale_demand(states, remaining)
    return demanded


def _calculate_positive_tests_to_distribute_per_age_group(
    n_newly_infected,
    share_known_cases,
    positivity_rate_overall,
    test_shares_by_age_group,
    positivity_rate_by_age_group,
):
    """Calculate how many positive test results each age group gets.

    Note this ignores inaccuracy of tests (false positives and negatives).

    Args:
        n_newly_infected (int): number of newly infected individuals.
        share_known_cases (float): share of infections that is detected.
        positivity_rate_overall (float): share of total tests that was positive.
        test_shares_by_age_group (pandas.Series): share of tests that was administered
            to each age group.
        positivity_rate_by_age_group (pandas.Series): share of tests that was positive
            in each age group.

    Returns:
        n_pos_tests_for_age_group (pandas.Series): number of positive tests
            to distribute in each age group.

    """
    n_pos_tests_overall = n_newly_infected * share_known_cases
    n_tests_overall = n_pos_tests_overall / positivity_rate_overall
    n_tests_for_each_group = n_tests_overall * test_shares_by_age_group
    n_pos_tests_for_each_group = n_tests_for_each_group * positivity_rate_by_age_group
    n_pos_tests_for_each_group = n_pos_tests_for_each_group.astype(int)
    return n_pos_tests_for_each_group


def _up_or_downscale_demand(states, remaining):
    demanded = states["demanded"].copy(deep=True)

    for group, remainder in remaining.items():
        n_to_draw = int(abs(remainder))
        selection_string = (
            f"age_group_rki == '{group}' & newly_infected & ~pending_test "
            + f"& ~knows_immune & demanded == {remainder < 0}"
        )
        pool = states.query(selection_string).index
        if len(pool) >= n_to_draw:
            drawn = np.random.choice(pool, n_to_draw, replace=False)
        else:
            warnings.warn(
                "There were more tests to be allocated / removed. "
                f"The remainder was {remainder} in group {group} on "
                f"{get_date(states).date()}."
            )
            drawn = pool
        demanded.loc[drawn] = True if remainder > 0 else False
    return demanded


def allocate_tests(n_allocated_tests, demands_test, states, params):  # noqa: U100
    """Allocate tests to individuals who demand a test.

    Excess and insufficient demand are handled in the demand function,
    so this is the identity function.

    Args:
        n_allocated_tests (int): The number of individuals who already
            received a test in this period from previous allocation models.
        demands_test (pandas.Series): A series with boolean entries
            where ``True`` indicates individuals asking for a test.
        states (pandas.DataFrame): The states of the individuals.
        params (pandas.DataFrame): A DataFrame with parameters.

    Returns:
        allocated_tests (numpy.ndarray, pandas.Series): An array or a
            series which indicates which individuals received a test.

    """
    return demands_test.copy(deep=True)


def process_tests(n_to_be_processed_tests, states, params):  # noqa: U100
    """Process tests.

    For simplicity, we assume that all tests are processed immediately, without
    further delay and without a capacity constraint.

    Args:
        n_to_be_processed_tests (int): Number of individuals whose test is
            already set to be processed.
        states (pandas.DataFrame): The states of the individuals.
        params (pandas.DataFrame): A DataFrame with parameters.

    Returns:
        started_processing (numpy.ndarray, pandas.Series): An array or series
            with boolean entries indicating which tests started to be processed.

    """
    to_be_processed_tests = states["pending_test"].copy(deep=True)
    return to_be_processed_tests
