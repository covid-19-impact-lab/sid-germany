import dask.dataframe as dd
import numpy as np
import pandas as pd


DEFAULT_WINDOW = 7
DEFAULT_TAKE_LOGS = True
DEFAULT_CENTER = False
DEFAULT_MIN_PERIODS = 1


def smoothed_outcome_per_hundred_thousand_sim(
    df,
    outcome,
    groupby=None,
    window=DEFAULT_WINDOW,
    min_periods=DEFAULT_MIN_PERIODS,
    take_logs=DEFAULT_TAKE_LOGS,
    center=DEFAULT_CENTER,
):
    """Calculated a smoothed outcome on the per 100 000 people level on simulated data.

    Args:
        df (pandas.DataFrame or dask.dataframe): Simulated time series.
        outcome (str): Selects a column in df.
        groupby (list, str or None): Defines the subgroups for which the outcome is
            calculated.
        window (int): Over how many days results are averaged to smooth the outcome.
        min_periods (int): Minimum number of days that need to be present in the
            smoothing window for the outcome to be not NaN.
        take_logs (int): Whether the log of the outcome should be returned. If True,
            smoothing is already done in logs.
        center (bool): Whether the smoothing window is centered or forward looking.

    Returns:
        pd.Series: Series with a smoothed outcome. The first index level is date. If
            groupby is specified, there are additional index levels.

    """
    df = df.reset_index()
    window, min_periods, groupby = _process_inputs(window, min_periods, groupby)
    per_individual = (
        df.groupby([pd.Grouper(key="date", freq="D")] + groupby)[outcome]
        .mean()
        .fillna(0)
    )

    if isinstance(df, dd.core.DataFrame):
        per_individual = per_individual.compute()

    out = _smooth_and_scale_daily_outcome_per_individual(
        per_individual, window, min_periods, groupby, take_logs, center=center
    )
    return out


def calculate_period_outcome_sim(df, outcome, groupby=None):
    """Calculate an outcome on a dataset of one period.

    This uses a groupby over the date column such that the date is preserved as the
    first index level of the result. Only meant to be during the msm estimation.

    Args:
        df (pandas.DataFrame): Simulated states DataFrame for one period.
        outcome (str): Selects a column in df.
        groupby (list, str or None): Defines the subgroups for which the outcome is
            calculated.

    Returns:
        pd.Series: Series with an unsmoothed outcome for one day. The first index
            level is date, even though it is meant to be the same for all entries.
            If groupby is specified, there are additional index levels.

    """
    if groupby is None:
        groupby = []
    elif isinstance(groupby, str):
        groupby = [groupby]

    out = (
        df.groupby([pd.Grouper(key="date", freq="D")] + groupby)[outcome]
        .mean()
        .fillna(0)
    )

    if isinstance(df, dd.core.DataFrame):
        out = out.compute()

    return out


def aggregate_and_smooth_period_outcome_sim(
    simulate_result,
    outcome,
    groupby=None,
    window=DEFAULT_WINDOW,
    min_periods=DEFAULT_MIN_PERIODS,
    take_logs=DEFAULT_TAKE_LOGS,
    center=DEFAULT_CENTER,
):
    """Aggregate and smooth a list of per period outcomes in simulate_results.

    Args:
        simulate_results (dict): Dictionary with a "period_outputs" entry.
        outcome (str): The name of the outcome in simulate_result["period_outputs"]
            that should be selected.
        groupby (list, str or None): Defines the subgroups for which the outcome is
            calculated.
        window (int): Over how many days results are averaged to smooth the outcome.
        min_periods (int): Minimum number of days that need to be present in the
            smoothing window for the outcome to be not NaN.
        take_logs (int): Whether the log of the outcome should be returned. If True,
            smoothing is already done in logs.
        center (bool): Whether the smoothing window is centered or forward looking.


    Returns:
        pd.Series: Series with a smoothed outcome. The first index level is date. If
            groupby is specified, there are additional index levels.

    """
    period_outcomes = simulate_result["period_outputs"][outcome]
    per_individual = pd.concat(period_outcomes)

    out = _smooth_and_scale_daily_outcome_per_individual(
        per_individual, window, min_periods, groupby, take_logs, center=center
    )
    return out


def smoothed_outcome_per_hundred_thousand_rki(
    df,
    outcome,
    groupby=None,
    window=DEFAULT_WINDOW,
    min_periods=DEFAULT_MIN_PERIODS,
    group_sizes=None,
    take_logs=DEFAULT_TAKE_LOGS,
    center=DEFAULT_CENTER,
):
    """Calculated a smoothed outcome on the per 100 000 people level on empirical data.

    Args:
        df (pandas.DataFrame): Empirical dataset.
        outcome (str): Selects a column in df.
        groupby (list, str or None): Defines the subgroups for which the outcome is
            calculated.
        window (int): Over how many days results are averaged to smooth the outcome.
        min_periods (int): Minimum number of days that need to be present in the
            smoothing window for the outcome to be not NaN.
        take_logs (int): Whether the log of the outcome should be returned. If True,
            smoothing is already done in logs.
        center (bool): Whether the smoothing window is centered or forward looking.

    Returns:
        pd.Series: Series with a smoothed outcome. The first index level is date. If
            groupby is specified, there are additional index levels.

    """
    df = df.reset_index()
    window, min_periods, groupby = _process_inputs(window, min_periods, groupby)

    per_individual = (
        df.groupby([pd.Grouper(key="date", freq="D")] + groupby)[outcome]
        .sum()
        .fillna(0)
    )
    if groupby:
        assert group_sizes is not None

    if not groupby:
        per_individual = per_individual / 83_000_000
    else:
        unstacked = per_individual.unstack()
        assert sorted(unstacked.columns) == sorted(group_sizes.index)
        per_individual = (unstacked / group_sizes).stack()

    out = _smooth_and_scale_daily_outcome_per_individual(
        per_individual,
        window,
        min_periods,
        groupby,
        take_logs,
        center=center,
    )
    return out


def _smooth_and_scale_daily_outcome_per_individual(
    sr,
    window,
    min_periods,
    groupby,
    take_logs,
    center,
):
    scaling_factor = 100_000
    scaled = sr * scaling_factor
    if take_logs:
        scaled = scaled.clip(1)
        scaled = np.log(scaled)
    if groupby:
        if not isinstance(scaled, (pd.Series, pd.DataFrame)):
            scaled = scaled.compute()
        scaled = scaled.unstack()
    smoothed = scaled.rolling(
        window=window, min_periods=min_periods, center=center
    ).mean()

    if groupby:
        smoothed = smoothed.stack()
    return smoothed


def _process_inputs(window, min_periods, groupby):
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")

    min_periods = int(min_periods)
    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")

    if groupby is None:
        groupby = []
    elif isinstance(groupby, str):
        groupby = [groupby]
    else:
        raise ValueError("groupby must be None or a string.")

    return window, min_periods, groupby
