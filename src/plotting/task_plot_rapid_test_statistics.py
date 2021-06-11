import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_SIZE
from src.plotting.plotting import BLUE
from src.plotting.plotting import RED
from src.plotting.plotting import style_plot
from src.simulation.scenario_config import create_path_to_rapid_test_statistics
from src.simulation.scenario_config import get_named_scenarios

SEEDS = get_named_scenarios()["combined_baseline"]["n_seeds"]

_DEPENDENCIES = {
    seed: create_path_to_rapid_test_statistics("combined_baseline", seed)
    for seed in range(SEEDS)
}


@pytask.mark.skipif(SEEDS == 0, reason="combined_baseline did not run.")
@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.produces(
    {
        "demand_shares": BLD
        / "figures"
        / "rapid_test_statistics"
        / "demand_shares.pdf",
        "share_infected_among_demand": BLD
        / "figures"
        / "rapid_test_statistics"
        / "share_infected_among_demand.pdf",
        "demand_shares_with_single_runs": BLD
        / "figures"
        / "rapid_test_statistics"
        / "demand_shares_with_single_runs.pdf",
        "share_infected_among_demand_with_single_runs": BLD
        / "figures"
        / "rapid_test_statistics"
        / "share_infected_among_demand_with_single_runs.pdf",
    }
)
def task_plot_rapid_test_statistics(depends_on, produces):
    dfs = {
        seed: pd.read_csv(path, parse_dates=["date"], index_col="date")
        for seed, path in depends_on.items()
    }
    for df in dfs.values():
        assert not df.index.duplicated().any(), (
            "Duplicates in a rapid test statistic DataFrame's index. "
            "The csv file must be deleted before every run."
        )

    demand_share_cols = [
        "private_demand_share",
        "work_demand_share",
        "educ_demand_share",
        "hh_demand",
        "sym_without_pcr_demand",
        "other_contact_demand",
    ]
    fig = _plot_columns(dfs, demand_share_cols, BLUE, False)
    fig.savefig(produces["demand_shares"])

    fig = _plot_columns(dfs, demand_share_cols, BLUE, True)
    fig.savefig(produces["demand_shares_with_single_runs"])

    share_infected_cols = [
        "share_infected_among_private_demand",
        "share_infected_among_work_demand",
        "share_infected_among_educ_demand",
        "share_infected_among_hh_demand",
        "share_infected_among_sym_without_pcr_demand",
        "share_infected_among_other_contact_demand",
    ]

    fig = _plot_columns(dfs, share_infected_cols, RED, False)
    fig.savefig(produces["share_infected_among_demand"])

    fig = _plot_columns(dfs, share_infected_cols, BLUE, True)
    fig.savefig(produces["share_infected_among_demand_with_single_runs"])

    plt.close()


def _plot_columns(dfs, cols, color, plot_single_runs):
    n_rows = int(np.ceil(len(cols) / 2))
    fig, axes = plt.subplots(
        ncols=2, nrows=n_rows, figsize=(PLOT_SIZE[0] * n_rows, PLOT_SIZE[1] * 2)
    )
    for col, ax in zip(cols, axes.flatten()):
        mean = pd.concat([df[col] for df in dfs.values()], axis=1).mean(axis=1)
        smoothed_mean = mean.rolling(window=7, min_periods=1, center=False).mean()
        sns.lineplot(
            x=smoothed_mean.index,
            y=smoothed_mean,
            ax=ax,
            linewidth=4,
            color=color,
            alpha=0.8,
        )

        if plot_single_runs:
            for df in dfs.values():
                sns.lineplot(
                    x=df.index,
                    y=df[col].rolling(window=7, min_periods=1, center=False).mean(),
                    ax=ax,
                    linewidth=2.5,
                    color=color,
                    alpha=0.6,
                )

        ax.set_title(col.replace("_", " ").title())
        fig, ax = style_plot(fig, ax)

    fig.tight_layout()
    return fig
