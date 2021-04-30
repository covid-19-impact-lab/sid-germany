import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.simulation.plotting import style_plot


def get_date_from_year_and_week(row):
    """Create date from year and week.

    We take the Sunday of each week.

    """
    date = datetime.date.fromisocalendar(
        year=int(row["year"]), week=int(row["week"]), day=7
    )
    return pd.Timestamp(date)


def convert_weekly_to_daily(df, divide_by_7_cols):
    """Convert from a weekly to a daily index.

    Each week is filled with the observation of the end of the week.
    Together with `get_date_from_year_and_week` taking the Sunday of
    each week, this yields the week's values for Mon through Sun to be
    the values of reported for that week.

    Args:
        df (pandas.DataFrame): DataFrame with
        divide_by_7_cols (list): list of columns that have to be
            divided by 7. So for example the number of participating
            laboratories does not change from a weekly to daily
            representation of the data but the available number of
            tests on each day is (ignoring weekends) a seventh of the
            weekly capacity.

    """
    dates = pd.date_range(df["date"].min() - pd.Timedelta(days=6), df["date"].max())
    df = df.set_index("date")
    df = df.reindex(dates)
    df = df.fillna(method="backfill")
    df = df.reset_index()
    df = df.rename(columns={"index": "date"})
    df[divide_by_7_cols] = df[divide_by_7_cols] / 7
    return df


def plot_time_series(df, y, title=""):
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.lineplot(data=df, x="date", y=y)
    ax.set_title(title)
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax


def get_piecewise_linear_interpolation_for_one_day(date, params_slice):
    """Get a linearly interpolated share known cases for one day.

    Args:
        date (pandas.Timestamp): Date at which the function is evaluated.
        params_slice (pandas.Series): Series with DateIndex. The values are
            function values corresponding to that date.

    Returns:
        float: The value of the function at that date.


    """
    date = pd.Timestamp(date)
    s = get_piecewise_linear_interpolation(params_slice)
    value = s.loc[date]
    return value


def get_piecewise_linear_interpolation(params_slice):
    """Get a linearly interpolated share known cases series."""
    if isinstance(params_slice, pd.DataFrame):
        params_slice = params_slice["value"]
    params_slice.index = pd.DatetimeIndex(params_slice.index)
    start_date = params_slice.index.min()
    end_date = params_slice.index.max()
    out = params_slice.reindex(pd.date_range(start_date, end_date)).interpolate()
    return out
