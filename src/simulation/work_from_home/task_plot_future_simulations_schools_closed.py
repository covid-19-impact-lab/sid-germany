import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.simulation.plotting import plot_incidences
from src.simulation.plotting import weekly_incidences_from_results
from src.simulation.work_from_home.task_simulate_future_schools_closed import (
    FUTURE_WFH_SEEDS,
)


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


FUTURE_WFH_SCENARIO_NAMES = [
    "mobility_data_baseline",
    "1st_lockdown_strict",
    "full_potential",
]

WFH_FUTURE_SIMULATION_PATHS = [
    BLD
    / "simulations"
    / "work_from_home_future_schools_closed"
    / f"{name}_{seed}"
    / "time_series"
    for name in FUTURE_WFH_SCENARIO_NAMES
    for seed in FUTURE_WFH_SEEDS
]

reported_series = {
    name: BLD
    / "simulations"
    / "work_from_home_future_schools_closed"
    / f"{name}_reported.pkl"
    for name in FUTURE_WFH_SCENARIO_NAMES
}

all_infected_series = {
    name: BLD
    / "simulations"
    / "work_from_home_future_schools_closed"
    / f"{name}_all_infected.pkl"
    for name in FUTURE_WFH_SCENARIO_NAMES
}


WFH_PLOT_PARAMETRIZATION = [
    (
        "new_known_case",
        "Beobachtete Inzidenz",
        {
            "fig": BLD
            / "simulations"
            / "work_from_home_future_schools_closed"
            / "reported.png",
            **reported_series,
        },
    ),
    (
        "newly_infected",
        "Tatsächliche Inzidenz",
        {
            "fig": BLD
            / "simulations"
            / "work_from_home_future_schools_closed"
            / "all_infected.png",
            **all_infected_series,
        },
    ),
]


@pytask.mark.skip
@pytask.mark.depends_on(
    WFH_FUTURE_SIMULATION_PATHS + [SRC / "simulation" / "plotting.py"]
)
@pytask.mark.parametrize("outcome, title, produces", WFH_PLOT_PARAMETRIZATION)
def task_plot_work_from_home_schools_closed_future_simulations(
    depends_on, outcome, title, produces
):
    # load simulation runs
    results = {}
    for name in FUTURE_WFH_SCENARIO_NAMES:
        result_paths = [path for path in depends_on.values() if name in str(path)]
        results[name] = [dd.read_parquet(path) for path in result_paths]

    # calculate incidences
    incidences = {}
    for name, simulation_runs in results.items():
        time_series_df = weekly_incidences_from_results(simulation_runs, outcome)
        time_series_df.mean(axis=1).to_pickle(produces[name])
        incidences[name] = time_series_df

    fig, ax = plot_incidences(incidences, 15, title)
    if title == "Beobachtete Inzidenz":
        ax.set_ylim(0, 200)
    ax.set_xlim(pd.Timestamp("2021-01-04"), pd.Timestamp("2021-03-02"))
    ax.set_xlabel("")

    fig.savefig(produces["fig"], dpi=200, transparent=False, facecolor="w")
