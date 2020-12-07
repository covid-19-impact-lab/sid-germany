import numpy as np
import pandas as pd
from sid.shared import boolean_choices


def create_initial_immunity(
    empirical_data,
    synthetic_data,
    initial_infections,
    undetected_multiplier,
    population_size,
    date,
    seed,
):
    """Create a Series with initial immunity.

    Args:
        empirical_data (pandas.Series): Newly infected Series with the index levels
            ["date", "county", "age_group_rki"].
        synthetic_data (pandas.DataFrame): Dataset with one row per simulated
            individual. Must contain the columns age_group_rki and county.
        initial_infections (pandas.DataFrame): DataFrame with same index as
            synthetic_data and one column for each day until *date*.
            Dtype is boolean.
        undetected_multiplier (float): Multiplier used to scale up the observed
            infections to account for unknown cases. Must be >=1.
        seed (int)

    Returns:
        pd.Series: Boolean series with same index as synthetic_data

    """
    empirical_data = empirical_data[: pd.Timestamp(date)].sort_index()

    initial_before_date = [
        pd.Timestamp(col) <= pd.Timestamp(date) for col in initial_infections
    ]
    assert all(initial_before_date), f"Initial infections must lie before {date}."
    assert undetected_multiplier >= 1, "undetected_multiplier must be >= 1."
    index_cols = ["date", "county", "age_group_rki"]
    right_index = empirical_data.index.names == index_cols
    assert right_index, f"Your data must have {index_cols} as index levels."
    duplicates_in_index = empirical_data.index.duplicated().any()
    assert not duplicates_in_index, "Your index must not have any duplicates."

    endog_immune = initial_infections.any(axis=1)
    total_immune = empirical_data.groupby(["age_group_rki", "county"]).sum()

    total_immunity_prob = _calculate_total_immunity_prob(
        total_immune,
        synthetic_data,
        undetected_multiplier,
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


def _calculate_total_immunity_prob(
    total_immunity, synthetic_data, undetected_multiplier, population_size
):
    """Calculate the probability to be immune by county and age group.

    Args:
        total_immunity (pandas.Series): index are the county and age group.
            Values are the total numbers of immune individuals.
        synthetic_data (pandas.DataFrame): DataFrame of synthetic individuals.
            Must contain age_group_rki and county as columns.
        undetected_multiplier (float): number of undetected infections per
            detected infection
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
    immunity_prob = (total_immunity / upscaled_group_sizes) * undetected_multiplier
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
