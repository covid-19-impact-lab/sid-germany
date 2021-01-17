import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(
    {
        "old": SRC
        / "original_data"
        / "testing"
        / "detected_and_undetected_infections.csv",
        "new": SRC
        / "original_data"
        / "testing"
        / "detected_and_undetected_infections_new.csv",
    }
)
@pytask.mark.produces(
    {
        "share_known_cases_until_christmas": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases_until_christmas.pkl",
        "share_known_cases_since_christmas": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases_since_christmas.pkl",
        "share_known_cases_fig": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases.png",
    }
)
def task_calculate_and_plot_share_known_cases(depends_on, produces):
    df_old = pd.read_csv(depends_on["old"])
    old_share_known = _calculate_share_known_cases(df_old)
    share_known_until_christmas = old_share_known[:"2020-12-24"]
    share_known_until_christmas.to_pickle(produces["share_known_cases_until_christmas"])

    df_new = pd.read_csv(depends_on["new"])
    new_share_known = _calculate_share_known_cases(df_new)
    share_known_since_christmas = new_share_known["2021-01-01":]
    share_known_since_christmas.to_pickle(produces["share_known_cases_since_christmas"])

    share_known_cases = pd.concat(
        [share_known_until_christmas, share_known_since_christmas]
    )

    fig, ax = _plot_time_series(share_known_cases, title="Share of Known Cases")
    fig.savefig(produces["share_known_cases_fig"])


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
    future_dates = pd.date_range(
        start=last_available_date, end=extrapolation_end_date, closed="right"
    )
    extrapolated_into_future = pd.Series(last_share, index=future_dates)

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
