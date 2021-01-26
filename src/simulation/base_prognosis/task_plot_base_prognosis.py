import dask.dataframe as dd
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
DEPENDENCIES = {
    "plotting_py": SRC / "simulation" / "plotting.py",
    "rki_data": BLD / "data" / "processed_time_series" / "rki.pkl",
}
for multiplier, seeds_and_paths in NESTED_PARAMETRIZATION.items():
    for seed, path in seeds_and_paths:
        DEPENDENCIES[(multiplier, seed)] = path

PLOT_PARAMETRIZATION = []
for outcome, title in [
    ("new_known_case", "Beobachtete Inzidenz"),
    ("newly_infected", "Tatsächliche Inzidenz"),
]:
    produces = {"fig": BLD / "simulations" / "baseline_prognosis" / f"{outcome}.png"}
    if outcome == "new_known_case":
        produces["data"] = (
            BLD / "simulations" / "baseline_prognosis" / "scenario_means.csv"
        )
    spec = (outcome, title, produces)
    PLOT_PARAMETRIZATION.append(spec)


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.parametrize("outcome, title, produces", PLOT_PARAMETRIZATION)
def task_plot_baseline_prognosis(depends_on, outcome, title, produces):
    # load simulation runs
    results = {}
    other_multipliers = NESTED_PARAMETRIZATION.keys()
    for multiplier in other_multipliers:
        result_paths = [val for key, val in depends_on.items() if key[0] == multiplier]
        results[multiplier] = [dd.read_parquet(path) for path in result_paths]

    # calculate incidences
    incidences = {}
    for multiplier, simulation_runs in results.items():
        time_series_df = weekly_incidences_from_results(simulation_runs, outcome)
        incidences[multiplier] = time_series_df

    if "data" in produces.keys():
        to_concat = [
            pd.Series(df.mean(axis=1), name=multiplier)
            for multiplier, df in incidences.items()
        ]
        means = pd.concat(to_concat, axis=1)
        means.to_csv(produces["data"])

    fig, ax = plot_incidences(incidences, 15, title, rki=outcome)

    fig.savefig(produces["fig"], dpi=200, transparent=False, facecolor="w")
