"""Testing models, adjusted from Tobi's sid tutorial.

We only model positive tests and assume there are no false positives or false negatives.
Note this assumes that individuals' behavior is unaffected by a negative test result.

This is very advantageous because only PCR tests are reported and antigen tests are not.
Thus, since positive antigen tests are followed up with a PCR test, positive antigen
tests show up in the test statistics and negative tests don't. So the positive tests
reflect the true positive tests but the negative tests don't.

Who gets a test as follows is completely determined in the demand_test function:

Firstly, we calculate from the number of infected people in the simulation and the
share_known_cases from the DunkelzifferRadar project how many positive tests are to
be distributed in the whole population. From this, using the overall positivity rate
of tests we get to the full budget of tests to be distributed across the population.
Using the ARS data, we get the share of tests (positive and negative) going to each
age group. Using the age specific positivity rate - also reported in the ARS data -
then gets us the number of positive tests to distribute in each age group.
Using the RKI and ARS data therefore allows us to reflect the German testing strategy
over age groups, e.g .preferential testing of older individuals.

We assume that symptomatic individuals preferentially demand and receive tests.
The remaining tests are distributed uniformly among the infectious in each age group.
We plan to further enhance the testing demand model by further variables such as contact
tracing.


"""
import warnings

import numpy as np
import pandas as pd
from sid.time import get_date


def demand_test(
    states,
    params,  # noqa: U100
    seed,  # noqa: U100
    share_known_cases,
    positivity_rate_overall,
    test_shares_by_age_group,
    positivity_rate_by_age_group,
):
    """Test demand function.

    This demand model already includes allocation and processing.

    We calculate the tests available in each age group as follows:
    Firstly, we calculate from the number of infected people in the simulation and the
    share_known_cases from the DunkelzifferRadar project how many positive tests are to
    be distributed in the whole population. From this, using the overall positivity rate
    of tests we get to the full budget of tests to be distributed across the population.
    Using the ARS data, we get the share of tests (positive and negative) going to each
    age group. Using the age specific positivity rate - also reported in the ARS data -
    then gets us the number of positive tests to distribute in each age group.
    Using the RKI and ARS data therefore allows us to reflect the German testing
    strategy over age groups, e.g .preferential testing of older individuals.

    In each age group we first distribute tests among those that are symptomatic but
    have no pending test and do not know their infection state yet. We then distribute
    the remaining tests tests among the remaining currently infectious such that we
    use up the full test budget in each group.

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
    demanded = states["symptomatic"] & ~states["pending_test"] & ~states["knows_immune"]
    demands_by_age_group = demanded.groupby(states["age_group_rki"]).sum()
    remaining = n_pos_tests_for_each_group - demands_by_age_group
    demanded = _scale_demand_up_or_down(demanded, states, remaining)
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


def _scale_demand_up_or_down(demanded, states, remaining):
    """Adjust the demand for tests to match the available tests in each age group.

    After symptomatic individuals have preferentially received tests the budget for
    tests in each age group may not be used up yet or exceeded. Here we remove the
    excess tests in the age groups where they exceed the budget. In groups were not
    all tests are used for symptomatic individuals we distribute the tests among the
    remaining infectious individuals that have no pending test and do not know their
    infection state yet.

    Args:
        demanded (pandas.Series): index is the same as that of states. It's boolean
            indicating whether the individual demands a test this period.
        states (pandas.DataFrame): sid states DataFrame
        remaining (pandas.Series): index are the RKI age groups, values are the
            number of remaining tests (can be negative) in each age group.

    Returns:
        demanded (pandas.Series): index is the same as that of states. It's boolean
            indicating whether the individual demands a test this period. The
            number of tests in each age group have been adjusted to match the number
            of available tests in that age group.

    """
    demanded = demanded.copy(deep=True)
    for group, remainder in remaining.items():
        n_to_draw = int(abs(remainder))
        selection_string = f"age_group_rki == '{group}' & ~pending_test & ~knows_immune"
        if remainder == 0:
            continue
        elif remainder > 0:
            # this is the case where we have additional positive tests to distribute.
            selection_string += " & infectious"
            pool = states[~demanded].query(selection_string).index
        else:  # remainder < 0
            # this is the case where symptomatics already exceed the available
            # number of positive tests.
            pool = states[demanded].query(selection_string).index
            warnings.warn(
                f"The demand for tests by symptomatic individuals in age group {group} "
                f"exceeded the number of available tests on {get_date(states).date()}. "
                "Conisder changing the model parameters such that you generalte less "
                "infected individuals or less infected individuals become symptomatic."
                f"There were {demanded.sum()} tests demanded "
                f"which was {-remainder} above the number of available tests."
            )
        if len(pool) >= n_to_draw:
            drawn = np.random.choice(pool, n_to_draw, replace=False)
        else:
            type_of_operation = "allocated" if remainder > 0 else "removed"
            warnings.warn(
                f"There were more tests to be {type_of_operation} than individuals "
                "available for this. This indicates that your model parameters "
                "(either the infection probabilities, the probability to become "
                "symptomatic or the test demand parameters are not well chosen. "
                f"The remainder was {remainder} in group {group} on "
                f"{get_date(states).date()}."
            )
            drawn = pool
        demanded.loc[drawn] = True if remainder > 0 else False
    return demanded


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
