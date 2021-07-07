import itertools

import pandas as pd
from sid.rapid_tests import _sample_test_outcome

from src.config import POPULATION_GERMANY
from src.policies.policy_tools import combine_dictionaries


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
    statistics = {
        "date": date,
    }

    demand_by_channel = demand_by_channel.copy()
    demand_by_channel["overall"] = demand_by_channel.any(axis=1)

    for channel in demand_by_channel.columns:
        # because we don't know the seed with which sample_test_outcome will be called
        # with, these results will not be exactly equal to the test outcomes in sid but
        # most channels easily exceed the number of tests for randomness to be relevant
        rapid_test_results = _sample_test_outcome(
            states=states,
            receives_rapid_test=demand_by_channel[channel],
            params=params,
            seed=itertools.count(93894),
        )

        channel_statistics = _calculate_rapid_test_statistics_by_channel(
            states=states,
            rapid_test_results=rapid_test_results,
            receives_rapid_test=demand_by_channel[channel],
            channel_name=channel,
        )
        statistics = combine_dictionaries([statistics, channel_statistics])

    statistics = pd.Series(statistics).to_frame()
    statistics.index.name = "index"
    return statistics


def _calculate_rapid_test_statistics_by_channel(
    states,
    rapid_test_results,
    receives_rapid_test,
    channel_name,
):
    """Calculate the rapid test statistics for one channel or overall.

    Naming convention for the denominators:

    - testshare             -> n_tests
    - popshare              -> n_people
    - number                -> n_people / POPULATION_GERMANY
    - rate                  -> n_{pos/neg}_tests

    Args:
        states (pandas.DataFrame): sid states DataFrame.
        receives_rapid_test (pandas.Series): boolean Series with the same index as
            states. This is the demand Series for one channel or overall.
        rapid_test_result (pandas.Series): boolean Series with the result for each
            individual. This is False for individuals that were not tested.
        channel_name (str): name of the channel.

    Returns:
        dict

    """
    tested_positive = rapid_test_results[receives_rapid_test]
    tested_negative = ~rapid_test_results[receives_rapid_test]
    n_tested = receives_rapid_test.sum()

    individual_outcomes = {
        "tested": receives_rapid_test,
        "tested_positive": tested_positive,
        "tested_negative": tested_negative,
        "true_positive": tested_positive & states["currently_infected"],
        "true_negative": tested_negative & ~states["currently_infected"],
        "false_positive": tested_positive & ~states["currently_infected"],
        "false_negative": tested_negative & states["currently_infected"],
    }

    statistics = {}
    for name, sr in individual_outcomes.items():
        statistics[f"number_{name}_by_{channel_name}"] = POPULATION_GERMANY * sr.mean()
        statistics[f"popshare_{name}_by_{channel_name}"] = sr.mean()
        if name != "tested":
            statistics[f"testshare_{name}_by_{channel_name}"] = sr.sum() / n_tested

    statistics[f"true_positive_rate_by_{channel_name}"] = (
        individual_outcomes["true_positive"].sum() / tested_positive.sum()
    )
    statistics[f"true_negative_rate_by_{channel_name}"] = (
        individual_outcomes["true_negative"].sum() / tested_negative.sum()
    )
    statistics[f"false_positive_rate_by_{channel_name}"] = (
        individual_outcomes["false_positive"].sum() / tested_positive.sum()
    )
    statistics[f"false_negative_rate_by_{channel_name}"] = (
        individual_outcomes["false_negative"].sum() / tested_negative.sum()
    )

    return statistics
