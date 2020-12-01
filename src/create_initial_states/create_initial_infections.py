import numpy as np
import pandas as pd
from sid.shared import boolean_choices


def create_initial_infections(
    empirical_data, synthetic_data, start, end, undetected_multiplier, seed
):
    """Create a DataFrame with initial infections.

    Args:
        empirical_data (pandas.DataFrame): Dataset with the columns age_group_rki,
            county, date and newly_infected. If there is more than one row per
            (county, date, age_group_rki) the newly_infected cases will be summed
            up over these groups.
        synthetic_data (pandas.DataFrame): Dataset with one row per simulated
            individual. Must contain the columns age_group_rki and county.
        start (str or pd.Timestamp): Start date.
        end (str or pd.Timestamp): End date.
        undetected_multiplier (float): Multiplier used to scale up the observed
            infections to account for unknown cases. Must be >=1.
        seed (int)

    Returns:
        pandas.DataFrame: DataFrame with same index as synthetic_data and one column
            for each day between start and end. Dtype is boolean.

    """
    np.random.seed(seed)
    assert undetected_multiplier >= 1, "undetected_multiplier must be >= 1."
    if (
        empirical_data.set_index(["date", "county", "age_group_rki"])
        .index.duplicated()
        .any()
    ):
        grouped = empirical_data.groupby(["date", "county", "age_group_rki"]).copy(
            deep=True
        )
        empirical_data = grouped.sum().fillna(0).reset_index()
    assert (
        empirical_data[["date", "county", "age_group_rki", "newly_infected"]]
        .notnull()
        .all()
        .all()
    ), "No NaN allowed in the empirical data"

    cases = _create_cases(empirical_data, start, end)
    infection_probs = _calculate_infection_probs(
        synthetic_data, cases, undetected_multiplier
    )

    initial_infections = pd.DataFrame(index=synthetic_data.index)
    for col in infection_probs:
        initial_infections[col] = boolean_choices(infection_probs[col].to_numpy())
    return initial_infections


def _create_cases(empirical_data, start, end):
    """Create a DataFrame of cases with dates as columns and age group and county as index.

    Args:
        empirical_data (pandas.DataFrame): Dataset with the columns age_group_rki,
            county date and newly_infected. There is one row per date, county and
            age group.
        start (str or pd.Timestamp): Start date.
        end (str or pd.Timestamp): End date.

    Returns:
        cases (pandas.DataFrame): DataFrame of cases with dates as columns and
            age group and county as index levels.

    """
    index_cols = ["date", "county", "age_group_rki"]
    cases = empirical_data.set_index(index_cols)["newly_infected"].copy(deep=True)
    cases = cases[pd.Timestamp(start) : pd.Timestamp(end)]  # noqa
    cases = cases.to_frame().unstack("date")
    cases.columns = [str(x.date()) for x in cases.columns.droplevel()]
    return cases


def _calculate_infection_probs(synthetic_data, cases, undetected_multiplier):
    """Calculate the infection probabilities from the cases and synthetic data.

    Args:
        synthetic_data (pandas.DataFrame): Dataset with one row per simulated
            individual. Must contain the columns age_group_rki and county.
        cases (pandas.DataFrame): DataFrame of cases with dates as columns and
            age group and county as index levels.
        undetected_multiplier (float): Multiplier used to scale up the observed
            infections to account for unknown cases. Must be >=1.

    Returns:
        infection_probs (pandas.DataFrame): Columns are the days given by cases.
            The index is the same as in synthetic data. The values are the
            probabilities for each individual to be infected on the particular day.

    """
    scaling_factor = 83_000_000 / len(synthetic_data)
    group_probability = (
        scaling_factor * synthetic_data.groupby(["county", "age_group_rki"]).size()
    )
    group_infection_probs = pd.DataFrame(index=cases.index)
    group_infection_probs["group_probability"] = group_probability

    for col in cases.columns:
        group_infection_probs[col] = (
            undetected_multiplier
            * cases[col]
            / group_infection_probs["group_probability"]
        )

    infection_probs = synthetic_data[["county", "age_group_rki"]].copy(deep=True)
    infection_probs = infection_probs.merge(
        group_infection_probs, on=["county", "age_group_rki"], validate="m:1"
    )
    infection_probs = infection_probs.drop(columns=["county", "age_group_rki"])
    return infection_probs
