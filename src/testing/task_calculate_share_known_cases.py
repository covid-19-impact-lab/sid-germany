import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(
    SRC / "original_data" / "testing" / "detected_and_undetected_infections.csv"
)
@pytask.mark.produces(
    {
        "share_known_cases": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases.pkl",
        "undetected_multiplier": BLD
        / "data"
        / "processed_time_series"
        / "undetected_multiplier.pkl",
        "share_known_cases_fig": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases.png",
        "undetected_multiplier_fig": BLD
        / "data"
        / "processed_time_series"
        / "undetected_multiplier.png",
    }
)
def task_calculate_and_plot_share_known_cases(depends_on, produces):
    df = pd.read_csv(depends_on)

    share_known_cases = _calculate_share_known_cases(df)
    share_known_cases.to_pickle(produces["share_known_cases"])
    undetected_multiplier = 1 / share_known_cases
    undetected_multiplier.to_pickle(produces["undetected_multiplier"])

    fig, ax = _plot_time_series(
        share_known_cases, title="Share of Known Cases in All Cases"
    )
    fig.savefig(produces["share_known_cases_fig"])
    undetected_title = (
        "Infection to Reported Cases Ratio\n"
        + "(also Known as Dark Figure or Undetected Multiplier)"
    )
    fig, ax = _plot_time_series(undetected_multiplier, undetected_title)
    fig.savefig(produces["undetected_multiplier_fig"])


def _calculate_share_known_cases(df):
    """Calculate the share of known cases from detected and undetected cases.

    Args:
        df (pandas.DataFrame): Dataframe with columns "date", "type" and "count".
            Each date and type is a row.

    Returns:
        share_known_cases (pandas.Series):
            share of known cases in the total number of cases.

    """
    df = df.set_index(["date", "type"]).unstack("type")
    df.columns = [x[1] for x in df.columns]
    df.index = pd.DatetimeIndex(df.index)
    df = df.rename(columns={"gemeldet": "known", "ungemeldet": "undetected"})
    df["total"] = df.sum(axis=1)
    share_known_cases = df["known"] / df["total"]
    min_share_until_june = share_known_cases[: pd.Timestamp("2020-06-01")].min()
    start_date = share_known_cases.index.min()
    jan_until_start = pd.date_range(start="2020-01-01", end=start_date, closed="left")
    extrapolated_share_before_start = pd.Series(
        min_share_until_june, index=jan_until_start
    )

    last_available_date = share_known_cases.index.max()
    last_share = share_known_cases[last_available_date]
    extrapolation_end_date = last_available_date + pd.Timedelta(weeks=8)
    six_weeks_into_future = pd.date_range(
        start=last_available_date, end=extrapolation_end_date, closed="right"
    )
    extrapolated_into_future = pd.Series(last_share, index=six_weeks_into_future)

    to_concat = [
        extrapolated_share_before_start,
        share_known_cases,
        extrapolated_into_future,
    ]
    share_known_cases = pd.concat(to_concat).sort_index()
    assert not share_known_cases.index.duplicated().any()
    assert (
        share_known_cases.index
        == pd.date_range(start="2020-01-01", end=extrapolation_end_date)
    ).all()
    return share_known_cases


def _plot_time_series(sr, title):
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.lineplot(x=sr.index, y=sr, ax=ax)
    ax.set_title(title)
    sns.despine()
    fig.tight_layout()
    return fig, ax
