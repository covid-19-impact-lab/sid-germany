import numpy as np
import pandas as pd
from sid.shared import boolean_choices

from src.config import POPULATION_GERMANY


def create_initial_immunity(
    empirical_data,
    synthetic_data,
    initial_infections,
    undetected_multiplier,
    date,
    seed,
    reporting_delay=0,
    population_size=POPULATION_GERMANY,
):
    """Create a Series with initial immunity.

    Args:
        empirical_data (pandas.Series): Newly infected Series with the index levels
            ["date", "county", "age_group_rki"].
        synthetic_data (pandas.DataFrame): Dataset with one row per simulated
            individual. Must contain the columns age_group_rki and county.
        initial_infections (pandas.DataFrame): DataFrame with same index as
            synthetic_data and one column for each day until *date*.
            Dtype is boolean. It is assumed that these already include
            undetected cases.
        undetected_multiplier (float or pandas.Series):
            Multiplier used to scale up the observed infections to account for
            undetected cases. Must be >=1.
        seed (int)
        reporting_delay (int): Number of days by which the reporting of cases is
            delayed. If given, later days are used to get the infections of the
            demanded time frame.
        population_size (int): Size of the population behind the empirical_data.

    Returns:
        pd.Series: Boolean series with same index as synthetic_data.

    """
    date_with_delay = pd.Timestamp(date) + pd.Timedelta(days=reporting_delay)
    empirical_data = empirical_data[:date_with_delay].sort_index()
    empirical_data.name = "reported_cases"

    initial_before_date = [
        pd.Timestamp(col) <= date_with_delay for col in initial_infections
    ]
    assert all(initial_before_date), f"Initial infections must lie before {date}."
    start = empirical_data.index.min()[0]

    if isinstance(undetected_multiplier, (float, int)):
        undetected_multiplier = pd.Series(
            data=undetected_multiplier,
            index=pd.date_range(start=start, end=date_with_delay),
        )
    undetected_multiplier.name = "undetected_multiplier"
    assert (undetected_multiplier >= 1).all(), "undetected_multiplier must be >= 1."
    assert set(pd.date_range(start, date_with_delay)).issubset(
        undetected_multiplier.index
    )

    index_cols = ["date", "county", "age_group_rki"]
    correct_index_levels = empirical_data.index.names == index_cols
    assert correct_index_levels, f"Your data must have {index_cols} as index levels."
    duplicates_in_index = empirical_data.index.duplicated().any()
    assert not duplicates_in_index, "Your index must not have any duplicates."

    endog_immune = initial_infections.any(axis=1)

    empirical_data = pd.merge(
        left=empirical_data.to_frame(),
        right=undetected_multiplier,
        left_on="date",
        right_index=True,
        how="left",
    )
    with_undetected_infections = (
        empirical_data["reported_cases"] * empirical_data["undetected_multiplier"]
    )
    total_immune = with_undetected_infections.groupby(["age_group_rki", "county"]).sum()

    total_immunity_prob = _calculate_total_immunity_prob(
        total_immune,
        synthetic_data,
        population_size,
    )
    endog_immunity_prob = _calculate_endog_immunity_prob(
        initial_infections,
        synthetic_data,
    )

    exog_immunity_prob = _calculate_exog_immunity_prob(
        total_immunity_prob, endog_immunity_prob
    )

    np.random.seed(seed)
    # need to duplicate exog prob on synthetical data
    hypothetical_exog_prob = pd.merge(
        synthetic_data,
        exog_immunity_prob,
        left_on=["age_group_rki", "county"],
        right_index=True,
        validate="m:1",
    )["exog_immunity_prob"]
    hypothetical_exog_prob = hypothetical_exog_prob.reindex(synthetic_data.index)

    hypothetical_exog_choice = pd.Series(
        boolean_choices(hypothetical_exog_prob.to_numpy()),
        index=synthetic_data.index,
    )
    return hypothetical_exog_choice.where(~endog_immune, endog_immune)


def _calculate_total_immunity_prob(total_immunity, synthetic_data, population_size):
    """Calculate the probability to be immune by county and age group.

    Args:
        total_immunity (pandas.Series): index are the county and age group.
            Values are the total numbers of immune individuals. These must
            already include undetected cases.
        synthetic_data (pandas.DataFrame): DataFrame of synthetic individuals.
            Must contain age_group_rki and county as columns.
        undetected_multiplier (pandas.Series): number of undetected infections per
            detected infection.
        population_size (int): number of individuals in the population from
            which the total_immunity was calculated.

    Returns:
        immunity_prob (pandas.Series): Index are county and age group
            combinations. Values are the probabilities of individuals of a
            particular county and age group to be immune.

    """
    upscale_factor = population_size / len(synthetic_data)
    synthetic_group_sizes = synthetic_data.groupby(["age_group_rki", "county"]).size()
    upscaled_group_sizes = synthetic_group_sizes * upscale_factor
    total_immunity = total_immunity.reindex(upscaled_group_sizes.index).fillna(0)
    immunity_prob = total_immunity / upscaled_group_sizes
    return immunity_prob


def _calculate_endog_immunity_prob(initial_infections, synthetic_data):
    """Calculate the immunity probability from initial infections.

    Args:
        initial_infections (pandas.DataFrame): DataFrame with same index as
            synthetic_data and one column for each day between start and end.
            Dtype is boolean.
        synthetic_data (pandas.DataFrame): Dataset with one row per simulated
            individual. Must contain the columns age_group_rki and county.

    Returns:
        prob_endog_immune (pandas.Series): Probabilities
            to become initially infected by age group and county.

    """
    df = synthetic_data[["age_group_rki", "county"]].copy()
    df["endog_immune"] = initial_infections.any(axis=1)
    prob_endog_immune = df.groupby(["age_group_rki", "county"])["endog_immune"].mean()
    return prob_endog_immune


def _calculate_exog_immunity_prob(total_immunity_prob, endog_immunity_prob):
    """Conditional probability to be immune, given not endogenously immune."""
    sr = (total_immunity_prob - endog_immunity_prob) / (1 - endog_immunity_prob)
    sr.name = "exog_immunity_prob"
    return sr
