import numpy as np
import pandas as pd


def smoothed_outcome_per_hundred_thousand_sim(
    df,
    outcome,
    groupby=None,
    window=14,
    min_periods=1,
    take_logs=True,
    center=True,
):
    df = df.reset_index()
    window, min_periods, groupby = _process_inputs(window, min_periods, groupby)
    per_individual = (
        df.groupby([pd.Grouper(key="date", freq="D")] + groupby)[outcome]
        .mean()
        .fillna(0)
    )

    if not isinstance(df, pd.DataFrame):
        per_individual = per_individual.compute()

    out = _smooth_and_scale_daily_outcome_per_individual(
        per_individual, window, min_periods, groupby, take_logs, center=center
    )
    return out


def smoothed_outcome_per_hundred_thousand_rki(
    df,
    outcome,
    groupby=None,
    window=7,
    min_periods=1,
    group_sizes=None,
    take_logs=True,
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
        scaled = scaled.compute().unstack()
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

    if not isinstance(groupby, (type(None), str)):
        raise ValueError("groupby must be a string.")

    groupby = [groupby] if groupby is not None else []

    return window, min_periods, groupby
