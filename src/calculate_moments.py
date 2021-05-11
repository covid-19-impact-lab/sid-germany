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
):
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
        per_individual, window, min_periods, groupby, take_logs
    )
    return out


def _smooth_and_scale_daily_outcome_per_individual(
    sr,
    window,
    min_periods,
    groupby,
    take_logs,
    center=True,
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
