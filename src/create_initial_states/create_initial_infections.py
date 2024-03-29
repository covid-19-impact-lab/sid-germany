import warnings

import numpy as np
import pandas as pd


def create_initial_infections(
    empirical_infections,
    synthetic_data,
    start,
    end,
    seed,
    virus_shares,
    reporting_delay,
    population_size,
):
    """Create a DataFrame with initial infections.

    .. warning::
        In case a person is drawn to be newly infected more than once we only
        infect her on the first date. If the probability of being infected is
        large, not correcting for this will lead to a lower infection probability
        than in the empirical data.

    Args:
        empirical_infections (pandas.Series): Newly infected Series with the index
            levels ["date", "county", "age_group_rki"]. Should already be corrected
            upwards to include undetected cases.
        synthetic_data (pandas.DataFrame): Dataset with one row per simulated
            individual. Must contain the columns age_group_rki and county.
        start (str or pd.Timestamp): Start date.
        end (str or pd.Timestamp): End date.
        seed (int)
        virus_shares (dict or None): If None, it is assumed that there is only one
            strain. If dict, keys are the names of the virus strains and the values
            are pandas.Series with a DatetimeIndex and the share among newly infected
            individuals on each day as value.
        reporting_delay (int): Number of days by which the reporting of cases is
            delayed. If given, later days are used to get the infections of the
            demanded time frame.
        population_size (int): Population size behind the empirical_infections.

    Returns:
        pandas.DataFrame: DataFrame with same index as synthetic_data and one column
            for each day between start and end. Dtype is boolean or categorical.
            Values identify which individual gets infected with which variant.

    """
    np.random.seed(seed)

    assert reporting_delay >= 0, "Reporting delay must be >= 0"
    reporting_delay = pd.Timedelta(days=reporting_delay)
    start = pd.Timestamp(start) + reporting_delay
    end = pd.Timestamp(end) + reporting_delay
    index_cols = ["date", "county", "age_group_rki"]
    correct_index_levels = empirical_infections.index.names == index_cols
    assert correct_index_levels, f"Your data must have {index_cols} as index levels."

    dates = empirical_infections.index.get_level_values("date").unique()
    expected_dates = pd.date_range(start, end)
    missing_dates = [str(x.date()) for x in expected_dates if x.date() not in dates]
    assert len(missing_dates) == 0, f"The following dates are missing: {missing_dates}"

    empirical_infections = empirical_infections.loc[
        pd.Timestamp(start) : pd.Timestamp(end)
    ]

    assert (
        empirical_infections.notnull().all().all()
    ), "No NaN allowed in the empirical data"
    duplicates_in_index = empirical_infections.index.duplicated().any()
    assert not duplicates_in_index, "Your index must not have any duplicates."

    cases = empirical_infections.to_frame().unstack("date")
    cases.columns = [str(x.date() - reporting_delay) for x in cases.columns.droplevel()]

    group_infection_probs = _calculate_group_infection_probs(
        cases, population_size, synthetic_data
    )

    initially_infected = _draw_bools_by_group(
        synthetic_data=synthetic_data,
        group_by=["county", "age_group_rki"],
        probabilities=group_infection_probs,
    )

    if virus_shares is not None:
        for sr in virus_shares.values():
            sr.index = sr.index - reporting_delay
        initially_infected = _add_variant_info_to_infections(
            initially_infected, virus_shares
        )

    return initially_infected


def _calculate_group_infection_probs(cases, population_size, synthetic_data):
    """Calculate the infection probability for each group and date.

    Args:
        cases (pandas.DataFrame): columns are the dates, the index are counties
            and age groups.
        population_size (int): Size of the population from which the cases
            originate.
        synthetic_data (pandas.DataFrame): Dataset with one row per simulated
            individual. Must contain the columns age_group_rki and county.

    Returns:
        group_infection_probs (pandas.DataFrame): columns are dates, index are
            counties and age groups. The values are the probabilities to be
            infected by age group on a particular date.

    """
    upscale_factor = population_size / len(synthetic_data)

    synthetic_group_sizes = synthetic_data.groupby(["county", "age_group_rki"]).size()
    upscaled_group_sizes = upscale_factor * synthetic_group_sizes
    cases = cases.reindex(upscaled_group_sizes.index).fillna(0)

    group_infection_probs = pd.DataFrame(index=upscaled_group_sizes.index)

    for col in cases.columns:
        prob = cases[col] / upscaled_group_sizes
        group_infection_probs[col] = prob

    return group_infection_probs


def _draw_bools_by_group(synthetic_data, group_by, probabilities):
    """Draw boolean values for each individual in synthetic data.

    Args:
        synthetic_data (pd.DataFrame): Synthetic data set containing
            the group_by variables.
        group_by (list): List of variables according to which the data
            are grouped.
        probabilities (pd.DataFrame): The index levels are the
            group_by variables. There can be several columns with
            probabilities.

    Returns:
        pandas.DataFrame or pandas.Series

    """
    group_indices = synthetic_data.groupby(group_by).groups
    res = pd.DataFrame(False, columns=probabilities.columns, index=synthetic_data.index)
    for group, indices in group_indices.items():
        group_size = len(indices)
        cases = pd.Series(
            _unbiased_sum_preserving_round(
                probabilities.loc[group].to_numpy() * group_size
            ),
            index=probabilities.columns,
        ).astype(int)
        remaining_indices = set(indices)
        for col, n_cases in cases.items():
            if len(remaining_indices) > 0:
                if n_cases > 0:
                    chosen = np.random.choice(
                        list(remaining_indices),
                        size=min(n_cases, len(remaining_indices)),
                        replace=False,
                    )
                    res.loc[chosen, col] = True
                    remaining_indices = remaining_indices - set(chosen)
            else:
                warnings.warn(
                    f"Every member of group {group} has been infected during the "
                    "burn in phase. If this happened with debug states, you can ignore "
                    "it, else you should investigate your estimates for the share of "
                    "known cases."
                )

    return res


def _unbiased_sum_preserving_round(arr):
    """Round values in an array, preserving the sum as good as possible.
    The function loops over the elements of an array and collects the deviations to the
    nearest downward adjusted integer. Whenever the collected deviations reach a
    predefined threshold, +1 is added to the current element and the collected
    deviations are reduced by 1.
    Args:
        arr (numpy.ndarray): 1d numpy array.
    Returns:
        numpy.ndarray
    """
    arr = arr.copy()

    threshold = np.random.uniform()
    deviation = 0

    for i in range(len(arr)):

        floor_value = int(arr[i])
        deviation += arr[i] - floor_value

        if deviation >= threshold:
            arr[i] = floor_value + 1
            deviation -= 1

        else:
            arr[i] = floor_value

    return arr


def _add_variant_info_to_infections(bool_df, virus_shares):
    """Draw which infections are of which virus variant.

    Args:
        bool_df (pandas.DataFrame): DataFrame with same index as
            synthetic_data and one column for each day between start and end.
            True for individuals being infected on each day.
        virus_shares (dict): A mapping between the names
            of the virus strains and their share among newly infected
            individuals over time.

    Returns:
        virus_strain_infections (pandas.DataFrame): DataFrame with same index as
            synthetic_data and one column for each day between start and end.
            Dtype is categorical, identifying which individual gets infected on
            each day by which variant.

    """
    virus_strain_infections = pd.DataFrame()
    names = sorted(virus_shares.keys())
    for date in bool_df:
        n_infections = bool_df[date].sum()
        strain_probs = [virus_shares[v_name][date] for v_name in names]
        sampled_strains = np.random.choice(a=names, p=strain_probs, size=n_infections)
        strain_infections = bool_df[date].replace({False: pd.NA, True: sampled_strains})
        virus_strain_infections[date] = pd.Categorical(
            strain_infections, categories=names
        )
    return virus_strain_infections
