import numpy as np
import pandas as pd
from sid.shared import boolean_choices

from src.config import POPULATION_GERMANY


def create_initial_infections(
    empirical_data,
    synthetic_data,
    start,
    end,
    undetected_multiplier,
    seed,
    population_size=POPULATION_GERMANY,
):
    """Create a DataFrame with initial infections.

    .. warning::
        In case a person is drawn to be newly infected more than once we only
        infect her on the first date. If the probability of being infected is
        large, not correcting for this will lead to a lower infection probability
        than in the empirical data.

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
        population_size (int): Size of the population behind the empirical_data.
        seed (int)

    Returns:
        pandas.DataFrame: DataFrame with same index as synthetic_data and one column
            for each day between start and end. Dtype is boolean.

    """
    np.random.seed(seed)

    assert undetected_multiplier >= 1, "undetected_multiplier must be >= 1."
    index_cols = ["date", "county", "age_group_rki"]
    if empirical_data.set_index(index_cols).index.duplicated().any():
        grouped = empirical_data.groupby().copy(deep=True)
        empirical_data = grouped.sum().fillna(0).reset_index()
    assert (
        empirical_data[index_cols + ["newly_infected"]].notnull().all().all()
    ), "No NaN allowed in the empirical data"

    cases = _create_cases(empirical_data, start, end)
    infection_probs = _calculate_infection_probs(
        synthetic_data, cases, undetected_multiplier, population_size
    )

    initial_infections = pd.DataFrame(index=synthetic_data.index)
    for col in infection_probs:
        initial_infections[col] = boolean_choices(infection_probs[col].to_numpy())

    initial_infections = _only_leave_first_true(initial_infections)
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


def _calculate_infection_probs(
    synthetic_data, cases, undetected_multiplier, population_size
):
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
    scaling_factor = population_size / len(synthetic_data)

    group_sizes = synthetic_data.groupby(["county", "age_group_rki"]).size()
    scaled_up_group_sizes = scaling_factor * group_sizes
    group_infection_probs = pd.DataFrame(index=cases.index)
    group_infection_probs["scaled_up_group_sizes"] = scaled_up_group_sizes

    for col in cases.columns:
        true_cases = undetected_multiplier * cases[col]
        prob = true_cases / group_infection_probs["scaled_up_group_sizes"]
        group_infection_probs[col] = prob

    infection_probs = synthetic_data[["county", "age_group_rki"]].copy(deep=True)
    infection_probs = infection_probs.merge(
        group_infection_probs, on=["county", "age_group_rki"], validate="m:1"
    )
    infection_probs = infection_probs.drop(
        columns=["county", "age_group_rki", "scaled_up_group_sizes"]
    )
    return infection_probs


def _only_leave_first_true(df):
    df = df.copy()
    for i, col in enumerate(df.columns):
        for other_col in df.columns[i + 1 :]:  # noqa
            df[other_col] = df[other_col] & ~df[col]
    return df
