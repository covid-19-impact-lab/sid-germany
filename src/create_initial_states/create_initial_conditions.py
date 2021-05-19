import itertools as it
import warnings

import pandas as pd

from src.config import POPULATION_GERMANY
from src.create_initial_states.create_initial_immunity import create_initial_immunity
from src.create_initial_states.create_initial_infections import (
    create_initial_infections,
)


def create_initial_conditions(
    start,
    end,
    seed,
    virus_shares,
    reporting_delay,
    synthetic_data,
    empirical_infections,
    population_size=POPULATION_GERMANY,
    overall_share_known_cases=None,
    group_share_known_cases=None,
    group_weights=None,
):
    """Create the initial conditions, initial_infections and initial_immunity.

    Args:
        start (str or pd.Timestamp): Start date for collection of initial
            infections.
        end (str or pd.Timestamp): End date for collection of initial
            infections and initial immunity.
        seed (int)
        virus_shares (dict): Keys are the names of the virus strains. Values are
            pandas.Series with a DatetimeIndex and the share among newly infected
            individuals on each day as value.
        reporting_delay (int): Number of days by which the reporting of cases is
            delayed. If given, later days are used to get the infections of the
            demanded time frame.
        synthetic_data (pandas.DataFrame): The synthetic population data set. Needs to
            contain 'county' and 'age_group_rki' as columns.
        empirical_infections (pandas.DataFrame): The index must contain 'date', 'county'
            and 'age_group_rki'.
        overall_share_known_cases (pd.Series): Series with date index that contains the
            aggregated share of known cases over time.
        group_share_known_cases (pandas.Series): Series with age_groups in the index.
            The values are interpreted as share of known cases for each age group.
        group_weights (pandas.Series): Series with sizes or weights of age groups.

    Returns:
        initial_conditions (dict): dictionary containing the initial infections and
            initial immunity.

    """
    seed = it.count(seed)
    upscaled_empirical_infections = _scale_up_empirical_new_infections(
        empirical_infections=empirical_infections,
        overall_share_known_cases=overall_share_known_cases,
        group_share_known_cases=group_share_known_cases,
        group_weights=group_weights,
    )

    initial_infections = create_initial_infections(
        empirical_infections=upscaled_empirical_infections,
        synthetic_data=synthetic_data,
        start=start,
        end=end,
        reporting_delay=reporting_delay,
        seed=next(seed),
        virus_shares=virus_shares,
        population_size=population_size,
    )

    initial_immunity = create_initial_immunity(
        empirical_infections=upscaled_empirical_infections,
        synthetic_data=synthetic_data,
        date=end,
        initial_infections=initial_infections,
        reporting_delay=reporting_delay,
        seed=next(seed),
        population_size=population_size,
    )
    return {
        "initial_infections": initial_infections,
        "initial_immunity": initial_immunity,
        # virus shares are already inside the initial infections so not included here.
    }


def _scale_up_empirical_new_infections(
    empirical_infections,
    group_share_known_cases=None,
    group_weights=None,
    overall_share_known_cases=None,
):
    """Scale up empirical infections with share of known cases.

    Args:
        empirical_infections (pandas.DataFrame): Must have the index levels date, county
            and age_group_rki and contain the column "newly_infected".
        group_share_known_cases (pandas.Series): Series with age_groups in the index.
            The values are interpreted as share of known cases for each age group.
        group_weights (pandas.Series): Series with sizes or weights of age groups.
        overall_share_known_cases (pd.Series): Series with date index that contains the
            aggregated share of known cases over time.

    Returns:
        pandas.Series: The upscaled new infections. Has the same index as
            empirical_infections.

    """
    if group_share_known_cases is not None:
        assert group_weights is not None

    dates = empirical_infections.index.get_level_values("date").unique()
    start = dates.min()
    end = dates.max()
    date_range = pd.date_range(start, end, name="date")
    group_weights = group_weights / group_weights.sum()

    if overall_share_known_cases is not None:
        overall_share_known_cases = (
            overall_share_known_cases.reindex(date_range)
            .fillna(method="bfill")
            .fillna(method="ffill")
        )

    group_share_known_cases_df = create_group_specific_share_known_cases(
        group_share_known_cases=group_share_known_cases,
        group_weights=group_weights,
        overall_share_known_cases=overall_share_known_cases,
        date_range=date_range,
    )
    stacked_group_share_known_cases = group_share_known_cases_df.stack()
    stacked_group_share_known_cases.name = "group_share_known_cases"

    if (stacked_group_share_known_cases > 0.95).any():
        stacked_group_share_known_cases = stacked_group_share_known_cases.clip(0, 0.95)
        warnings.warn(
            "The group specific share known cases is > 0.95 for some date and group. "
            "If this happened with debug states you can simply ignore it. If it "
            "happened with full states, you should investigate it. The group's share "
            "known cases has been clipped to 0.95.",
            UserWarning,
        )

    if (stacked_group_share_known_cases < 0.05).any():
        stacked_group_share_known_cases = stacked_group_share_known_cases.clip(0.05, 1)
        warnings.warn(
            "The group specific share known cases is < 0.05 for some date and group. "
            "If this happened with debug states you can simply ignore it. If it "
            "happened with full states, you should investigate it. The group's share "
            "known cases has been clipped to 0.05.",
            UserWarning,
        )

    merged = pd.merge(
        empirical_infections.reset_index(),
        right=stacked_group_share_known_cases.reset_index(),
        on=["date", "age_group_rki"],
    )

    merged["upscaled_newly_infected"] = (
        merged["newly_infected"] / merged["group_share_known_cases"]
    )
    merged = merged.set_index(["date", "county", "age_group_rki"])
    return merged["upscaled_newly_infected"]


def create_group_specific_share_known_cases(
    group_share_known_cases,
    group_weights,
    overall_share_known_cases,
    date_range,
):
    """Create the group specific share known cases.

    Args:
        group_share_known_cases (pandas.Series): Series with age_groups in the index.
            The values are interpreted as share of known cases for each age group.
        group_weights (pandas.Series): Series with sizes or weights of age groups.
        overall_share_known_cases (pd.Series): Series with date index that contains the
            aggregated share of known cases over time.


    Returns:
        pandas.DataFrame: The index are the dates, the columns are the group labels. The
            value is the share known cases of the particular group on the particular
            date.

    """
    age_groups = group_weights.index
    # None given
    if group_share_known_cases is None and overall_share_known_cases is None:
        raise ValueError("Either group or overall share_known_cases must be given.")
    # both given
    elif group_share_known_cases is not None and overall_share_known_cases is not None:
        implied_overall = group_share_known_cases @ group_weights
        scaling_factor = overall_share_known_cases / implied_overall
        group_share_known_cases_df = pd.DataFrame(
            data=[group_share_known_cases] * len(date_range), index=date_range
        )
        for col in group_share_known_cases_df:
            group_share_known_cases_df[col] = (
                group_share_known_cases_df[col] * scaling_factor
            )

    # only overall given
    elif overall_share_known_cases is not None:
        group_share_known_cases_df = pd.concat(
            [overall_share_known_cases] * len(age_groups), axis=1
        )
        group_share_known_cases_df.columns = age_groups
    # only group given
    else:
        group_share_known_cases_df = pd.DataFrame(
            data=[group_share_known_cases] * len(date_range), index=date_range
        )
    return group_share_known_cases_df
