import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.simulation.base_prognosis.base_prognosis_specification import (
    build_base_prognosis_parametrization,
)
from src.simulation.plotting import plot_incidences
from src.simulation.plotting import weekly_incidences_from_results

NESTED_PARAMETRIZATION = build_base_prognosis_parametrization()
PLOT_PARAMETRIZATION = []
for outcome, title in [
    ("new_known_case", "Beobachtete Inzidenz"),
    ("newly_infected", "Tatsächliche Inzidenz"),
]:
    for other_multiplier, seeds_and_paths in NESTED_PARAMETRIZATION.items():
        fig_name = f"{str(other_multiplier).replace('.', '_')}_{outcome}.png"
        out_path = BLD / "simulations" / "baseline_prognosis" / fig_name
        spec = (
            outcome,
            title,
            [path for seed, path in seeds_and_paths],
            out_path,
        )
        PLOT_PARAMETRIZATION.append(spec)

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


@pytask.mark.depends_on([SRC / "simulation" / "plotting.py"])
@pytask.mark.parametrize("outcome, title, depends_on, produces", PLOT_PARAMETRIZATION)
def task_plot_baseline_prognosis_simulations(outcome, title, depends_on, produces):
    # load simulation runs
    results = {}
    for name in []:
        result_paths = [path for path in depends_on.values() if name in str(path)]
        results[name] = [dd.read_parquet(path) for path in result_paths]

    # calculate incidences
    incidences = {}
    for name, simulation_runs in results.items():
        time_series_df = weekly_incidences_from_results(simulation_runs, outcome)
        incidences[name] = time_series_df

    fig, ax = plot_incidences(incidences, 15, title)
    if title == "Beobachtete Inzidenz":
        ax.set_ylim(0, 200)
    ax.set_xlim(pd.Timestamp("2021-01-04"), pd.Timestamp("2021-03-02"))
    ax.set_xlabel("")

    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")
