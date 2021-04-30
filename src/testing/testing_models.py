"""PCR testing model for sid."""
import warnings

import numpy as np
import pandas as pd
from sid.time import get_date

from src.testing.shared import get_piecewise_linear_interpolation_for_one_day


def demand_test(
    states,
    params,
    seed,
    share_of_tests_for_symptomatics_series,
):
    """Test demand function.

    Contrary to the name this function combines test demand and test allocation.

    Args:
        states (pandas.DataFrame): The states of the individuals.
        params (pandas.DataFrame): A DataFrame with parameters. It needs to contain
            the entry ("test_demand", "symptoms", "share_symptomatic_requesting_test").
        seed (int): Seed for reproducibility.
        share_of_tests_for_symptomatics_series (pandas.Series): Series with date index
            that indicates the share of positive tests that were allocated to
            symptomatic people.

    Returns:
        demand_probability (numpy.ndarray, pandas.Series): An array or a series
            which contains the probability for each individual demanding a test.

    """
    np.random.seed(seed)
    date = get_date(states)

    # extract parameters
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )

        loc = ("test_demand", "shares", "share_w_positive_rapid_test_requesting_test")
        share_requesting_confirmation = params.loc[loc, "value"]

        params_slice = params.loc[("share_known_cases", "share_known_cases")]

    share_known_cases = get_piecewise_linear_interpolation_for_one_day(
        date, params_slice
    )

    share_of_tests_for_symptomatics = share_of_tests_for_symptomatics_series[date]

    test_demand_from_share_known_cases = _calculate_test_demand_from_share_known_cases(
        states=states,
        share_known_cases=share_known_cases,
        share_of_tests_for_symptomatics=share_of_tests_for_symptomatics,
    )

    test_demand_from_rapid_tests = _calculate_test_demand_from_rapid_tests(
        states, share_requesting_confirmation
    )

    demanded = test_demand_from_share_known_cases | test_demand_from_rapid_tests

    return demanded


def _calculate_test_demand_from_share_known_cases(states, share_known_cases, share_of_tests_for_symptomatics):
    n_newly_infected = states["newly_infected"].sum()
    n_pos_tests = n_newly_infected * share_known_cases
    untested = ~states["knows_immune"] & ~states["pending_test"]

    symptomatic_untested = states["symptomatic"] & untested
    n_symptomatic_untested = symptomatic_untested.sum()

    desired_n_tests_symptomatic = n_pos_tests * share_of_tests_for_symptomatics
    n_tests_symptomatic = int(min(desired_n_tests_symptomatic, n_symptomatic_untested))

    n_tests_remaining = int(n_pos_tests - n_tests_symptomatic)

    symptomatic_pool = states.index[symptomatic_untested]
    symptomatic_sampled = np.random.choice(symptomatic_pool, size=n_tests_symptomatic, replace=False)

    is_remaining_candidate = states["currently_infected"] & ~states["symptomatic"] & untested
    remaining_pool = states.index[is_remaining_candidate]
    remaining_sampled = np.random.choice(remaining_pool, size=n_tests_remaining, replace=False)

    demand = pd.Series(False, index=states.index)

    for loc in [symptomatic_sampled, remaining_sampled]:
        demand[loc] = True

    return demand


def _calculate_test_demand_from_rapid_tests(states, share_requesting_confirmation):
    received_rapid_test = states["cd_received_rapid_test"] == 0
    pos_rapid_test = states["is_tested_positive_by_rapid_test"]
    pool = states[received_rapid_test & pos_rapid_test].index
    n_to_draw = int(share_requesting_confirmation * len(pool))
    demands_verification_locs = np.random.choice(
        a=pool,
        size=n_to_draw,
        replace=False,
    )
    demanding_verification = states.index.isin(demands_verification_locs)
    getting_confirmation = states["currently_infected"] & demanding_verification
    return getting_confirmation


def allocate_tests(n_allocated_tests, demands_test, states, params, seed):  # noqa: U100
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
    allocated_tests = demands_test.copy(deep=True)
    return allocated_tests


def process_tests(n_to_be_processed_tests, states, params, seed):  # noqa: U100
    """Process tests.

    For simplicity, we assume that all tests are processed immediately, without
    further delay and without a capacity constraint.

    When tests are processed, sid starts the test countdowns which we take from
    the RKI data (see https://tinyurl.com/2urakgwa for details) which reports
    the data from taking the test sample to notifying the subject of his/her
    result. This aligns well with our test demand function assigning test demand
    to symptomatic individuals and currently infectious individuals (starting
    with sid commit d9185a8).

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
