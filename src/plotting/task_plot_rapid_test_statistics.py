import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.plotting.plotting import BLUE
from src.plotting.plotting import RED
from src.plotting.plotting import style_plot
from src.simulation.scenario_config import create_path_to_rapid_test_statistics
from src.simulation.scenario_config import get_named_scenarios

SEEDS = get_named_scenarios()["combined_baseline"]["n_seeds"]

_CSV_DEPENDENCIES = {
    seed: create_path_to_rapid_test_statistics("combined_baseline", seed)
    for seed in range(SEEDS)
}

DEMAND_SHARE_COLS = [
    "private_demand_share",
    "work_demand_share",
    "educ_demand_share",
    "hh_demand",
    "sym_without_pcr_demand",
    "other_contact_demand",
]

SHARE_INFECTED_COLS = [
    "share_infected_among_private_demand",
    "share_infected_among_work_demand",
    "share_infected_among_educ_demand",
    "share_infected_among_hh_demand",
    "share_infected_among_sym_without_pcr_demand",
    "share_infected_among_other_contact_demand",
]

TABLE_PATH = BLD / "tables" / "rapid_test_statistics"


_CSV_PARAMETRIZATION = [
    (column, TABLE_PATH / f"{column}.csv")
    for column in DEMAND_SHARE_COLS + SHARE_INFECTED_COLS
]


@pytask.mark.skipif(SEEDS == 0, reason="combined_baseline did not run.")
@pytask.mark.depends_on(_CSV_DEPENDENCIES)
@pytask.mark.parametrize("column, produces", _CSV_PARAMETRIZATION)
def task_process_rapid_test_statistics(depends_on, column, produces):
    dfs = {
        seed: pd.read_csv(path, parse_dates=["date"], index_col="date")
        for seed, path in depends_on.items()
    }
    for df in dfs.values():
        assert not df.index.duplicated().any(), (
            "Duplicates in a rapid test statistic DataFrame's index. "
            "The csv file must be deleted before every run."
        )
    df = pd.concat({seed: df[column] for seed, df in dfs.items()}, axis=1)
    df["smoothed_mean"] = (
        df.mean(axis=1).rolling(window=7, min_periods=1, center=False).mean()
    )
    df.to_csv(produces)


_PLOT_PARAMETRIZATION = []
for column in DEMAND_SHARE_COLS:
    for plot_single_runs in [True, False]:
        depends_on = TABLE_PATH / f"{column}.csv"
        ylabel = "share of the population demanding a rapid test"
        file_name = (
            f"{column}_with_single_runs.pdf" if plot_single_runs else f"{column}.pdf"
        )
        produces = BLD / "figures" / "rapid_test_statistics" / file_name
        spec = (depends_on, BLUE, plot_single_runs, ylabel, produces)
        _PLOT_PARAMETRIZATION.append(spec)

for column in SHARE_INFECTED_COLS:
    for plot_single_runs in [True, False]:
        depends_on = TABLE_PATH / f"{column}.csv"
        ylabel = "share of rapid tests demanded by infected individuals"
        file_name = (
            f"{column}_with_single_runs.pdf" if plot_single_runs else f"{column}.pdf"
        )

        produces = BLD / "figures" / "rapid_test_statistics" / file_name
        spec = (depends_on, RED, plot_single_runs, ylabel, produces)
        _PLOT_PARAMETRIZATION.append(spec)


@pytask.mark.parametrize(
    "depends_on, color, plot_single_runs, ylabel, produces", _PLOT_PARAMETRIZATION
)
def task_plot_rapid_test_statistics(
    depends_on, color, plot_single_runs, ylabel, produces
):
    df = pd.read_csv(depends_on, index_col="date", parse_dates=["date"])
    fig = _plot_df(df=df, color=color, plot_single_runs=plot_single_runs, ylabel=ylabel)
    fig.savefig(produces)
    plt.close()


def _plot_df(df, color, plot_single_runs, ylabel):
    fig, ax = plt.subplots(figsize=PLOT_SIZE)

    sns.lineplot(
        x=df["smoothed_mean"].index,
        y=df["smoothed_mean"],
        ax=ax,
        linewidth=4,
        color=color,
        alpha=0.8,
    )

    if plot_single_runs:
        for col in df.columns:
            if col != "smoothed_mean":
                sns.lineplot(
                    x=df.index,
                    y=df[col].rolling(window=7, min_periods=1, center=False).mean(),
                    ax=ax,
                    linewidth=2.5,
                    color=color,
                    alpha=0.6,
                )
    ax.set_xlim(pd.Timestamp("2021-03-15"), pd.Timestamp(PLOT_END_DATE))
    fig, ax = style_plot(fig, ax)

    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig
