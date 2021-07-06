import itertools

import numpy as np
import pandas as pd
from sid.rapid_tests import _sample_test_outcome


def create_rapid_test_statistics(demand_by_channel, states, date, params):
    """Calculate the rapid test statistics.

    Args:
        demand_by_channel (pandas.DataFrame): same index as states. Each column is one
            channel through which rapid tests can be demanded.
        states (pandas.DataFrame): sid states DataFrame.
        date (pandas.Timestamp or str): date
        params (pandas.DataFrame): parameter DataFrame that contains the sensitivity
            and specificity of the rapid tests

    Returns:
        statistics (pandas.DataFrame): DataFrame with just one column named 0. The
            index contains the date, the number of individuals and for each channel
            the share of the population that demand a test through this channel, the
            share of tests in each channel that are demanded by infected individuals.

    """
    weights = _calculate_weights(demand_by_channel)

    statistics = {
        "date": date,
        "n_individuals": len(states),
    }

    for channel in demand_by_channel.columns:
        statistics[f"share_with_rapid_test_through_{channel}"] = weights[channel].mean()
        statistics[f"n_rapid_tests_through_{channel}"] = weights[channel].sum()
        statistics[
            f"share_of_{channel}_rapid_tests_demanded_by_infected"
        ] = _share_demanded_by_infected(
            demand_by_channel=demand_by_channel,
            states=states,
            weights=weights,
            channel=channel,
        )

        # because we don't know the seed with which sample_test_outcome will be called
        # with, these results will not be exactly equal to the test outcomes in sid but
        # most channels easily exceed the number of tests for randomness to be relevant
        rapid_test_results = _sample_test_outcome(
            states=states,
            receives_rapid_test=demand_by_channel[channel],
            params=params,
            seed=itertools.count(93894),
        )

        (
            share_true_positive,
            share_false_negative,
            _,
        ) = _calculate_true_positive_and_false_negatives(
            states=states,
            receives_rapid_test=demand_by_channel[channel],
            rapid_test_results=rapid_test_results,
        )
        statistics[f"true_positive_rate_in_{channel}"] = share_true_positive
        statistics[f"false_positive_rate_in_{channel}"] = 1 - share_true_positive
        statistics[f"true_negative_rate_in_{channel}"] = 1 - share_false_negative
        statistics[f"false_negative_rate_in_{channel}"] = share_false_negative

    # overall results
    receives_rapid_test = demand_by_channel.any(axis=1)
    statistics["share_with_rapid_test"] = receives_rapid_test.mean()
    statistics["n_rapid_tests_overall"] = receives_rapid_test.sum()

    # no need to worry about the impact of the seed. After Feb 2021 the number of tests
    # always 1000 every day and exceeds 100,000 per day after mid April
    rapid_test_results = _sample_test_outcome(
        states=states,
        receives_rapid_test=receives_rapid_test,
        params=params,
        seed=itertools.count(93894),
    )

    (
        share_true_positive,
        share_false_negative,
        share_false_positive_in_population,
    ) = _calculate_true_positive_and_false_negatives(
        states=states,
        receives_rapid_test=receives_rapid_test,
        rapid_test_results=rapid_test_results,
    )
    statistics["true_positive_rate_overall"] = share_true_positive
    statistics["false_positive_rate_overall"] = 1 - share_true_positive
    statistics["true_negative_rate_overall"] = 1 - share_false_negative
    statistics["false_negative_rate_overall"] = share_false_negative
    statistics[
        "false_positive_rate_in_the_population"
    ] = share_false_positive_in_population

    statistics = pd.Series(statistics).to_frame()
    statistics.index.name = "index"
    return statistics


def _share_demanded_by_infected(demand_by_channel, states, weights, channel):
    """Calculate the share of *channel* tests that are demanded by infected individuals.

    Args:
        demand_by_channel (pandas.DataFrame): each individual is a row, each column is
            a channel through which an individual may demand a rapid test. Cells are
            True if the individual demanded a rapid test through this channel.
        states (pandas.DataFrame): sid states DataFrame.
        weights (pandas.DataFrame): columns and index are the same as in
            demand_by_channel. Each cell is the share to which this individual demanded
            a test through the particular channel.
        channel (str): column name for which to calculate the share that is demanded by
            infected individuals.

    Returns:
        share_demanded_by_infected (float): share of the rapid tests of a channel that
            are demanded by infected individuals.

    """
    demanders = demand_by_channel[channel]
    infected_demanders = demanders & states["currently_infected"]
    total_demand = weights.loc[demanders, channel].sum()
    infected_demand = weights.loc[infected_demanders, channel].sum()
    if total_demand > 0:
        share_demanded_by_infected = infected_demand / total_demand
    else:
        share_demanded_by_infected = np.nan
    return share_demanded_by_infected


def _calculate_weights(demand_by_channel):
    """Calculate to which share

    Args:
        demand_by_channel (pandas.DataFrame): each individual is a row, each column is
            a channel through which an individual may demand a rapid test. Cells are
            True if the individual demanded a rapid test through this channel.

    Returns:
        weights (pandas.DataFrame): columns and index are the same as in
            demand_by_channel. Each cell is the share to which this individual demanded
            a test through the particular channel. For example, if an individual has a
            rapid test demand through both work and the household her weight in each of
            the two channels is 0.5. If an individual does not a rapid test because of a
            channel, her weight is 0.

    """
    weights = demand_by_channel.div(demand_by_channel.sum(axis=1), axis=0).fillna(0)
    return weights


def _calculate_true_positive_and_false_negatives(
    states, rapid_test_results, receives_rapid_test
):
    """Calculate the share that is tested true positive and false negative.

    Args:
        states (pandas.DataFrame): sid states DataFrame.
        receives_rapid_test (pandas.Series): boolean Series with the same index as
            states.
        rapid_test_result (pandas.Series): boolean Series with

    Returns:
        float: share of the positive tests that are given to infected individuals.
        float: share of the negative tests that are given to infected individuals.
        float: share of the general population receiving a false positive rapid test.

    """
    # reduce to the test takers
    test_takers = states[receives_rapid_test]
    tested_positive = rapid_test_results[receives_rapid_test]
    tested_negative = ~rapid_test_results[receives_rapid_test]

    share_true_positive = test_takers[tested_positive]["currently_infected"].mean()
    share_false_negative = test_takers[tested_negative]["currently_infected"].mean()

    false_positive_in_population = tested_positive & states["currently_infected"]
    share_false_positive_in_population = false_positive_in_population.mean()

    return share_true_positive, share_false_negative, share_false_positive_in_population
