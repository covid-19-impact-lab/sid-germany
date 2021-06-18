import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from sid.colors import get_colors

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.config import PLOT_START_DATE
from src.config import SRC
from src.plotting.plotting import style_plot

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


@pytask.mark.depends_on(
    {
        "hygiene": BLD / "policies" / "hygiene_score.csv",
        "work": BLD / "policies" / "work_multiplier.csv",
        "plotting.py": SRC / "plotting" / "plotting.py",
    }
)
@pytask.mark.produces(
    {
        "hygiene": BLD / "policies" / "hygiene_score.pdf",
        "by_state": BLD / "policies" / "work_mobility_reduction_by_state.pdf",
        "de": BLD / "policies" / "work_multiplier.pdf",
        "since_sep": BLD / "figures" / "data" / "work_multiplier_since_sep.pdf",
        "old_vs_new": BLD / "policies" / "old_vs_new_work_multipliers_since_sep.pdf",
        "2021": BLD / "policies" / "work_mobility_2021.pdf",
    }
)
def task_visualize_work_multipliers(depends_on, produces):
    hygiene_score = pd.read_csv(depends_on["hygiene"], parse_dates=["date"])
    work_multiplier = pd.read_csv(depends_on["work"], parse_dates=["date"])
    sns.set_palette(get_colors("categorical", 12))

    fig, ax = _plot_time_series(
        data=hygiene_score,
        y="hygiene_score",
        x="date",
        title="Hygiene Score Acc. to the COSMOS Data",
    )
    fig.savefig(produces["hygiene"])
    plt.close()

    fig, ax = _visualize_reductions_by_state(df=work_multiplier)
    fig.savefig(produces["by_state"])
    plt.close()

    fig, ax = _plot_time_series(work_multiplier, title="Work Multiplier")
    fig.savefig(produces["de"])
    plt.close()

    fig, ax = _plot_time_series(work_multiplier, title="Work Multiplier")
    fig.savefig(produces["de"])
    plt.close()

    since_sep = work_multiplier[
        work_multiplier["date"].between(PLOT_START_DATE, PLOT_END_DATE)
    ]
    fig, ax = _plot_time_series(since_sep, title="")
    ax.set_ylabel("mobility relative to pre-pandemic level")
    fig.savefig(produces["since_sep"])
    plt.close()

    old_multipliers = _get_old_work_multipliers()
    sns.lineplot(
        x=old_multipliers.index, y=old_multipliers, ax=ax, label="old multipliers"
    )
    fig.savefig(produces["old_vs_new"])
    plt.close()

    this_year = work_multiplier[work_multiplier["date"] > "2021-01-03"]
    fig, ax = _plot_time_series(this_year, title="Reduction of Work Mobility in 2021")
    plt.axvline(
        x=pd.Timestamp("2021-01-27"),
        label="Corona-Arbeitsschutzverordnung",
        color="firebrick",
    )
    plt.legend()
    fig.savefig(
        produces["2021"],
        dpi=200,
        transparent=False,
        facecolor="w",
    )
    plt.close()
    # Undo side effect of sns.set_palette above.
    sns.color_palette()


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
    data = data[data[x].between(PLOT_START_DATE, PLOT_END_DATE)]
    y = [y] if isinstance(y, str) else y
    if ax is None:
        fig, ax = plt.subplots(figsize=PLOT_SIZE)

    for var in y:
        sns.lineplot(data=data, y=var, x=x, ax=ax, label=var)

    fig, ax = style_plot(fig, ax)
    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_xlabel("")
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
