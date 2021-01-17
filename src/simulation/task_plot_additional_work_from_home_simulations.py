import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from matplotlib.dates import DateFormatter
from sid.colors import get_colors

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_sim
from src.config import BLD
from src.simulation.task_simulate_additional_work_from_home import WFH_SEEDS
from src.simulation.task_simulate_additional_work_from_home_future import (
    FUTURE_WFH_SCENARIO_NAMES,
)
from src.simulation.task_simulate_additional_work_from_home_future import (
    FUTURE_WFH_SEEDS,
)


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

WFH_SCENARIO_NAMES = [
    "baseline",
    "1st_lockdown_weak",
    "1st_lockdown_strict",
    "full_potential",
]


WFH_SIMULATION_PATHS = [
    BLD / "simulations" / "work_from_home" / f"{name}_{seed}" / "time_series"
    for name in WFH_SCENARIO_NAMES
    for seed in WFH_SEEDS
]

WFH_PLOT_PARAMETRIZATION = [
    (
        "new_known_case",
        "Beobachtete Inzidenz",
        BLD / "simulations" / "work_from_home" / "reported.png",
    ),
    (
        "newly_infected",
        "Tatsächliche Inzidenz",
        BLD / "simulations" / "work_from_home" / "all_infected.png",
    ),
]


@pytask.mark.skip
@pytask.mark.depends_on(WFH_SIMULATION_PATHS)
@pytask.mark.parametrize("outcome, title, produces", WFH_PLOT_PARAMETRIZATION)
def task_plot_work_from_home_simulations(depends_on, outcome, title, produces):
    # load simulation runs
    results = {}
    for name in WFH_SCENARIO_NAMES:
        result_paths = [path for path in depends_on.values() if name in str(path)]
        results[name] = [dd.read_parquet(path) for path in result_paths]

    # calculate incidences
    incidences = {}
    for name, simulation_runs in results.items():
        incidences[name] = _weekly_incidences_from_results(simulation_runs, outcome)

    fig, ax = _plot_incidences(incidences, 15, title)
    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")


WFH_FUTURE_SIMULATION_PATHS = [
    BLD / "simulations" / "work_from_home_future" / f"{name}_{seed}" / "time_series"
    for name in FUTURE_WFH_SCENARIO_NAMES
    for seed in FUTURE_WFH_SEEDS
]

WFH_PLOT_PARAMETRIZATION = [
    (
        "new_known_case",
        "Beobachtete Inzidenz",
        BLD / "simulations" / "work_from_home_future" / "reported.png",
    ),
    (
        "newly_infected",
        "Tatsächliche Inzidenz",
        BLD / "simulations" / "work_from_home_future" / "all_infected.png",
    ),
]


@pytask.mark.depends_on(WFH_FUTURE_SIMULATION_PATHS)
@pytask.mark.parametrize("outcome, title, produces", WFH_PLOT_PARAMETRIZATION)
def task_plot_work_from_home_future_simulations(depends_on, outcome, title, produces):
    # load simulation runs
    results = {}
    for name in FUTURE_WFH_SCENARIO_NAMES:
        result_paths = [path for path in depends_on.values() if name in str(path)]
        results[name] = [dd.read_parquet(path) for path in result_paths]

    # calculate incidences
    incidences = {}
    for name, simulation_runs in results.items():
        incidences[name] = _weekly_incidences_from_results(simulation_runs, outcome)

    fig, ax = _plot_incidences(incidences, 15, title)
    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")


def _weekly_incidences_from_results(results, outcome):
    """Create the weekly incidences from a list of simulation runs.

    Args:
        results (list): list of DataFrames with the

    Returns:
        weekly_incidences (pandas.DataFrame): every column is the
            weekly incidence over time for one simulation run.
            The index are the dates of the simulation period.

    """
    weekly_incidences = []
    for res in results:
        weekly_incidences.append(
            smoothed_outcome_per_hundred_thousand_sim(
                df=res,
                outcome=outcome,
                take_logs=False,
                window=7,
                center=False,
            )
            * 7
        )
    weekly_incidences = pd.concat(weekly_incidences, axis=1)
    weekly_incidences.columns = range(len(results))
    return weekly_incidences


def _plot_incidences(incidences, n_single_runs, title):
    """Plot incidences.

    Args:
        incidences (dict): keys are names of the scenarios,
            values are dataframes where each column is the
            incidence of interest of one run
        n_single_runs (int): number of individual runs to
            visualize to show statistical uncertainty.
        title (str): plot title.

    Returns:
        fig, ax

    """
    colors = get_colors("ordered", len(incidences))
    fig, ax = plt.subplots(figsize=(6, 4))
    name_to_label = {
        "baseline": "Tatsächliche Home Office-Quote (14%)",
        "1_pct_more": "1 Prozent Mehr Home Office",
        "1st_lockdown_weak": "Home Office wie im Frühjarhrslockdown, untere Grenze (25%)",  # noqa: E501
        "1st_lockdown_strict": "Home Office wie im Frühjarhrslockdown, obere Grenze (35%)",  # noqa: E501
        "full_potential": "Volles Ausreizen des Home Office Potentials (55%)",
    }
    for name, color in zip(incidences, colors):
        df = incidences[name]
        dates = df.index
        sns.lineplot(
            x=dates,
            y=df.mean(axis=1),
            ax=ax,
            color=color,
            label=name_to_label[name],
            linewidth=2.5,
            alpha=0.8,
        )
        # plot individual runs to visualize statistical uncertainty
        for run in df.columns[:n_single_runs]:
            sns.lineplot(
                x=dates,
                y=df[run],
                ax=ax,
                color=color,
                linewidth=0.5,
                alpha=0.2,
            )

    ax.set_ylabel("")
    ax.set_xlabel("Datum")
    date_form = DateFormatter("%d.%m")
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    ax.set_ylabel("Geglättete wöchentliche \nNeuinfektionen pro 100 000")
    ax.grid(axis="y")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(-0.1, -0.5, 1, 0.2))
    fig.tight_layout()

    return fig, ax
