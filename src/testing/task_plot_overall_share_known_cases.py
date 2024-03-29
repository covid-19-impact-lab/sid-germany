import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.config import PLOT_START_DATE
from src.config import SRC
from src.plotting.plotting import BLUE
from src.plotting.plotting import RED
from src.plotting.plotting import style_plot
from src.testing.shared import get_piecewise_linear_interpolation


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
        "params": BLD / "params.pkl",
        "plotting.py": SRC / "plotting" / "plotting.py",
        "testing_shared.py": SRC / "testing" / "shared.py",
    }
)
@pytask.mark.produces(
    {
        "share_known_cases_fig": BLD
        / "figures"
        / "data"
        / "testing"
        / "assumed_overall_share_known_cases.pdf",
    }
)
def task_plot_overall_share_known_cases(depends_on, produces):
    df_old = pd.read_csv(depends_on["old"])
    old_share_known = _calculate_share_known_cases(df_old)[PLOT_START_DATE:"2020-12-24"]

    df_new = pd.read_csv(depends_on["new"])
    new_share_known = _calculate_share_known_cases(df_new)["2020-12-25":]
    share_known = pd.concat([old_share_known, new_share_known])
    share_known.index = share_known.index.normalize()
    assert not share_known.index.duplicated().any()
    dates = share_known.index
    expected_dates = pd.date_range(dates.min(), dates.max())
    missing_dates = [str(x.date()) for x in expected_dates if x not in dates]
    assert (
        len(missing_dates) == 0
    ), f"There are missing dates in the share_known: {missing_dates}"

    share_known = share_known.loc[PLOT_START_DATE:"2020-12-28"]

    params = pd.read_pickle(depends_on["params"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        params_slice = params.loc[("share_known_cases", "share_known_cases")]
    share_known_from_params = get_piecewise_linear_interpolation(params_slice)
    share_known_from_params = share_known_from_params.loc[PLOT_START_DATE:PLOT_END_DATE]

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.lineplot(
        x=share_known.index,
        y=share_known,
        ax=ax,
        label="Dunkelzifferradar",
        alpha=0.6,
        linewidth=3.0,
        color=BLUE,
    )
    sns.lineplot(
        x=share_known_from_params.index,
        y=share_known_from_params,
        ax=ax,
        label="Interpolated and Extrapolated",
        alpha=0.6,
        linewidth=3.0,
        color=RED,
    )
    fig, ax = style_plot(fig, ax)
    ax.set_ylabel("share of cases that are detected")
    fig.tight_layout()

    ax.axvline(pd.Timestamp("2020-12-24"), alpha=0.6, color="k")
    fig.savefig(produces["share_known_cases_fig"])
    plt.close()


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
    extrapolation_end_date = last_available_date + pd.Timedelta(weeks=16)
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
