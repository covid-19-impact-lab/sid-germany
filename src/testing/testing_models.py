"""Testing models, adjusted from Tobi's sid tutorial.

We only model positive tests and assume there are no false positives or false negatives.
Note this assumes that individuals' behavior is unaffected by a negative test result.

This is very advantageous because only PCR tests are reported and antigen tests are not.
Thus, since positive antigen tests are followed up with a PCR test, positive antigen
tests show up in the test statistics and negative tests don't. So the positive tests
reflect the true positive tests but the negative tests don't.

Who gets a test as follows is completely determined in the demand_test function:

Firstly, we calculate from the number of infected people in the simulation and the
share_known_cases how many positive tests are to
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

from src.contact_models.contact_model_functions import get_states_w_vacations
from src.testing.shared import get_share_known_cases_for_one_day


def demand_test(
    states,
    params,
    seed,
    positivity_rate_overall,
    test_shares_by_age_group,
    positivity_rate_by_age_group,
):
    """Test demand function.

    Test demand is calculated in such a way that the demand fits to the empirical
    distribution of positive tests in the empirical data.

    We calculate the tests designated in each age group as follows:
    Firstly, we calculate from the number of infected people in the simulation and the
    share_known_cases how many positive tests are to
    be distributed in the whole population. From this, using the overall positivity rate
    of tests we get to the full budget of tests to be distributed across the population.
    Using the ARS data, we get the share of tests (positive and negative) going to each
    age group. Using the age specific positivity rate - also reported in the ARS data -
    then gets us the number of positive tests to distribute in each age group.
    Using the RKI and ARS data therefore allows us to reflect the German testing
    strategy over age groups, e.g .preferential testing of older individuals.

    In each age group we first distribute tests among those that recently developed
    symptoms but have no pending test and do not know their infection state yet.
    We then test all education workers such as teachers that have not been tested
    in the last week and are not on vacation.
    We then distribute the remaining tests among the remaining currently
    infectious such that we use up the full test budget in each age group.

    Args:
        states (pandas.DataFrame): The states of the individuals.
        params (pandas.DataFrame): A DataFrame with parameters. It needs to contain
            the entry ("test_demand", "symptoms", "share_symptomatic_requesting_test").
        seed (int): Seed for reproducibility.
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
    np.random.seed(seed)
    n_newly_infected = states["newly_infected"].sum()
    symptom_tuple = ("test_demand", "symptoms", "share_symptomatic_requesting_test")
    share_symptomatic_requesting_test = params.loc[symptom_tuple, "value"]
    date = get_date(states)
    if isinstance(test_shares_by_age_group, pd.DataFrame):
        test_shares_by_age_group = test_shares_by_age_group.loc[date]
    if isinstance(positivity_rate_by_age_group, pd.DataFrame):
        positivity_rate_by_age_group = positivity_rate_by_age_group.loc[date]
    if isinstance(positivity_rate_overall, pd.Series):
        positivity_rate_overall = positivity_rate_overall.loc[date]
    if share_symptomatic_requesting_test > 1.0 or share_symptomatic_requesting_test < 0:
        raise ValueError(
            "The share of symptomatic individuals requesting a test must lie in the "
            f"[0, 1] interval, you specified {share_symptomatic_requesting_test}"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        params_slice = params.loc[("share_known_cases", "share_known_cases")]
    share_known_cases = get_share_known_cases_for_one_day(date, params_slice)

    n_pos_tests_for_each_group = _calculate_positive_tests_to_distribute_per_age_group(
        n_newly_infected=n_newly_infected,
        share_known_cases=share_known_cases,
        positivity_rate_overall=positivity_rate_overall,
        test_shares_by_age_group=test_shares_by_age_group,
        positivity_rate_by_age_group=positivity_rate_by_age_group,
    )
    developed_symptoms_yesterday = states["cd_symptoms_true"] == -1
    symptomatic_without_test = (
        developed_symptoms_yesterday & ~states["pending_test"] & ~states["knows_immune"]
    )
    if share_symptomatic_requesting_test == 1.0:
        demanded = symptomatic_without_test
    else:
        # this ignores the designated number of tests per age group.
        # Adjusting the number of tests to the designated number is done in
        # `_scale_demand_up_or_down` below.
        n_to_demand = int(
            share_symptomatic_requesting_test * symptomatic_without_test.sum()
        )
        pool = states[symptomatic_without_test].index
        drawn = np.random.choice(size=n_to_demand, a=pool, replace=False)
        demanded = pd.Series(False, index=states.index)
        demanded[drawn] = True

    if date > pd.Timestamp("2020-12-31"):
        demanded = _demand_test_for_educ_workers(demanded, states, params)

    demands_by_age_group = demanded.groupby(states["age_group_rki"]).sum()
    remaining = n_pos_tests_for_each_group - demands_by_age_group
    demanded = _scale_demand_up_or_down(demanded, states, remaining)

    if (remaining < 0).any():
        info = pd.concat([demands_by_age_group, n_pos_tests_for_each_group], axis=1)
        info.columns = ["demand", "target demand"]
        info["difference"] = (info["demand"] - info["target demand"]) / info[
            "target demand"
        ]
        info = info.T.round(2)
        warnings.warn(
            f"Too much endogenous test demand on {date.date()} ({date.day_name()}). "
            "This is an indication that the share of symptomatic infections is too high or"
            f"that too many symptomatic people demand a test:\n\n{info.to_string()}"
        )

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


def _demand_test_for_educ_workers(demanded, states, params):
    """Every working teacher who has not received a test in the last 7 days is tested.

    At the moment we only distribute positive tests. As a result the
    `cd_received_test_result_true` countdown does not give us who has been tested and
    we only want to demand tests for education workers who will get a positive result.

    We implement the tests for education workers as spread out across the week. This
    is necessary because teachers use antigen tests for which no data is available.
    Our test data is weekly PCR test data which we spread out to be evenly distributed
    across the week. So we spread out the testing for teachers also across the week.
    We use the index to decide who gets tested which day of the week. Since the states
    are shuffeld this is not a problem.

    We again assume that tests are perfect, i.e. no false positives or negatives.

    """
    demanded = demanded.copy()
    date = get_date(states)
    states_w_vacations = get_states_w_vacations(date, params)
    on_vacation = states["state"].isin(states_w_vacations)
    working_teachers = states["educ_worker"] & ~on_vacation

    day_of_week = date.dayofweek
    slice_start = int(day_of_week / 7 * len(states))
    slice_end = int(((day_of_week + 1) / 7) * len(states))
    in_slice = (slice_start <= states["index"]) & (states["index"] < slice_end)
    to_be_tested = working_teachers & in_slice
    to_receive_positive_test = to_be_tested & states["infectious"]
    demanded[to_receive_positive_test] = True
    return demanded


def _scale_demand_up_or_down(demanded, states, remaining):
    """Adjust the demand for tests to match the designated tests in each age group.

    After symptomatic individuals have preferentially received tests the budget for
    tests in each age group may not be used up yet or exceeded. Here we remove the
    excess tests in the age groups where they exceed the budget. In groups were not
    all tests are used for symptomatic individuals we distribute the tests among the
    remaining infectious individuals that have no pending test and do not know their
    infection state yet.

    Args:
        demanded (pandas.Series): Boolean Series with same index as states. It is
            True for people who demanded a test.
        states (pandas.DataFrame): sid states DataFrame
        remaining (pandas.Series): index are the RKI age groups, values are the
            number of remaining tests (can be negative) in each age group.

    Returns:
        demanded (pandas.Series): Boolean Series with same index as states. It is
            True for people who demanded a test. The number of tests in each age
            group have been adjusted to match the number of designated tests in
            that age group.

    """
    demanded = demanded.copy(deep=True)
    for group, remainder in remaining.items():
        if remainder == 0:
            continue
        elif remainder > 0:
            n_undemanded_tests = int(abs(remainder))
            demanded = _increase_test_demand(
                demanded, states, n_undemanded_tests, group
            )
        else:  # remainder < 0
            n_to_remove = int(abs(remainder))
            demanded = _decrease_test_demand(demanded, states, n_to_remove, group)
    return demanded


def _decrease_test_demand(demanded, states, n_to_remove, group):
    """Decrease the number of tests demanded in an age group by a certain number.

    This is called when the endogenously demanded tests (symptomatics + educ workers)
    already exceed the designated number of positive tests in an age group.

    """
    demanded = demanded.copy(deep=True)
    age_str = f"age_group_rki == '{group}'"
    demanding_test_in_age_group = states[demanded].query(age_str).index
    if len(demanding_test_in_age_group) >= n_to_remove:
        drawn = np.random.choice(
            a=demanding_test_in_age_group, size=n_to_remove, replace=False
        )
        demanded.loc[drawn] = False
    else:
        raise ValueError(
            "Trying to remove more test demands than there are "
            f"individuals requesting a test in age group {group}."
        )
    return demanded


def _increase_test_demand(demanded, states, n_undemanded_tests, group):
    """Randomly increase the number of tests demanded in an age group.
    This is the case where we have additional positive tests to distribute.

    """
    demanded = demanded.copy(deep=True)
    age_query = f"(age_group_rki == '{group}')"
    infected_query = "(infectious | symptomatic | (cd_infectious_true >= 0))"
    untested_query = "(~pending_test & ~knows_immune)"
    selection_string = age_query + " & " + infected_query + " & " + untested_query
    infected_untested = states[~demanded].query(selection_string).index
    if len(infected_untested) >= n_undemanded_tests:
        drawn = np.random.choice(infected_untested, n_undemanded_tests, replace=False)
    else:
        date = get_date(states)
        warnings.warn(
            f"\n\nThe implied share_known_cases for age group {group} is >1 "
            f"on date {date.date()} ({date.day_name()}).\n\n"
        )
        drawn = infected_untested
    demanded.loc[drawn] = True
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
