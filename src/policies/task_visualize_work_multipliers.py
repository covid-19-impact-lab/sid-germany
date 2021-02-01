import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from matplotlib.dates import AutoDateLocator
from matplotlib.dates import DateFormatter
from sid.colors import get_colors

from src.config import BLD

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

sns.set_palette(get_colors("categorical", 12))


@pytask.mark.depends_on(
    {
        "hygiene": BLD / "policies" / "hygiene_score.csv",
        "work": BLD / "policies" / "work_multiplier.csv",
    }
)
@pytask.mark.produces(
    {
        "hygiene": BLD / "policies" / "hygiene_score.png",
        "by_state": BLD / "policies" / "work_mobility_reduction_by_state.png",
        "de": BLD / "policies" / "work_multiplier.png",
        "since_oct": BLD / "policies" / "work_multiplier_since_oct.png",
    }
)
def task_visualize_work_multipliers(depends_on, produces):
    hygiene_score = pd.read_csv(depends_on["hygiene"], parse_dates=["date"])
    work_multiplier = pd.read_csv(depends_on["work"], parse_dates=["date"])

    fig, ax = _plot_time_series(
        data=hygiene_score,
        y="hygiene_score",
        x="date",
        title="Hygiene Score Acc. to the COSMOS Data",
    )
    fig.savefig(produces["hygiene"], dpi=200, transparent=False, facecolor="w")

    fig, ax = _visualize_reductions_by_state(df=work_multiplier)
    fig.savefig(produces["by_state"], dpi=200, transparent=False, facecolor="w")

    fig, ax = _plot_time_series(work_multiplier, title="Work Multiplier")
    fig.savefig(produces["de"], dpi=200, transparent=False, facecolor="w")

    fig, ax = _plot_time_series(work_multiplier, title="Work Multiplier")
    fig.savefig(produces["de"], dpi=200, transparent=False, facecolor="w")

    since_oct = work_multiplier[work_multiplier["date"] > "2020-10-01"]
    fig, ax = _plot_time_series(since_oct, title="Work Multiplier Since October")
    old_multipliers = _get_old_work_multipliers()
    sns.lineplot(
        x=old_multipliers.index, y=old_multipliers, ax=ax, label="old multipliers"
    )
    fig.savefig(produces["since_oct"], dpi=200, transparent=False, facecolor="w")


def _visualize_reductions_by_state(df):
    states = ["Bavaria", "North Rhine-Westphalia", "Saxony", "Mecklenburg-Vorpommern"]
    fig, ax = _plot_time_series(data=df, y=states)
    title = "Work Multiplier Acc. to Google Mobility Data by State"
    ax.set_title(title)
    return fig, ax


def _plot_time_series(
    data,
    y="Germany",
    x="date",
    title="",
    fig=None,
    ax=None,
):
    y = [y] if isinstance(y, str) else y
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))

    for var in y:
        sns.lineplot(data=data, y=var, x=x, ax=ax, label=var)

    date_form = (
        DateFormatter("%d/%m/%Y") if len(data) < 100 else DateFormatter("%d/%m/%Y")
    )
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    loc = AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(loc)
    ax.grid(axis="y")
    ax.set_title(title)
    return fig, ax


def _get_old_work_multipliers():
    """Get the old work multipliers for the policy periods from oct to feb.

    Returns:
        out (pandas.Series): index are the dates from 2020-10-01 to 2021-02-01.
            Values are the old work multipliers

    """
    hygiene_multiplier = 0.966667  # compared to October
    work_multiplier = 0.693 * hygiene_multiplier

    to_combine = [
        {"start_date": "2020-10-01", "end_date": "2020-10-09", "work": 0.8415},
        {"start_date": "2020-10-10", "end_date": "2020-10-23", "work": 0.7458},
        {"start_date": "2020-10-24", "end_date": "2020-11-01", "work": 0.8415},
        {
            "start_date": "2020-11-02",
            "end_date": "2020-11-22",
            "work": 0.8118 * hygiene_multiplier,
        },
        {
            "start_date": "2020-11-23",
            "end_date": "2020-12-15",
            "work": 0.8316 * hygiene_multiplier,
        },
        {
            "start_date": "2020-12-16",
            "end_date": "2020-12-20",
            "work": 0.8316 * hygiene_multiplier,
        },
        {"start_date": "2020-12-21", "end_date": "2020-12-23", "work": 0.5},
        {"start_date": "2020-12-24", "end_date": "2020-12-26", "work": 0.3},
        {
            "start_date": "2020-12-27",
            "end_date": "2021-01-03",
            "work": 0.429,
        },
        {
            "start_date": "2021-01-04",
            "end_date": "2021-01-11",
            "work": 0.594 * hygiene_multiplier,
        },
        {
            "start_date": "2021-01-12",
            "end_date": "2021-02-01",
            "work": work_multiplier,
        },
    ]
    out = pd.Series(
        index=pd.date_range("2020-10-01", "2021-02-01"), name="old_work_multiplier"
    )
    for d in to_combine:
        out[d["start_date"] : d["end_date"]] = d["work"]  # noqa: E203
    return out
