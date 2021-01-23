import dask.dataframe as dd
import pytask

from src.config import BLD
from src.config import SRC
from src.simulation.base_prognosis.base_prognosis_specification import (
    build_base_prognosis_parametrization,
)
from src.simulation.plotting import plot_incidences
from src.simulation.plotting import weekly_incidences_from_results

NESTED_PARAMETRIZATION = build_base_prognosis_parametrization()
DEPENDENCIES = [SRC / "simulation" / "plotting.py"]
for seeds_and_paths in NESTED_PARAMETRIZATION.values():
    DEPENDENCIES += [path for seed, path in seeds_and_paths]

PLOT_PARAMETRIZATION = []
for outcome, title in [
    ("new_known_case", "Beobachtete Inzidenz"),
    ("newly_infected", "Tatsächliche Inzidenz"),
]:
    out_path = BLD / "simulations" / "baseline_prognosis" / f"{outcome}.png"
    spec = (outcome, title, out_path)
    PLOT_PARAMETRIZATION.append(spec)


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.parametrize("outcome, title, produces", PLOT_PARAMETRIZATION)
def task_plot_baseline_prognosis(depends_on, outcome, title, produces):
    # load simulation runs
    results = {}
    other_multipliers = NESTED_PARAMETRIZATION.keys()
    for multiplier in other_multipliers:
        flag_str = str(multiplier).replace(".", "_")
        result_paths = [path for path in depends_on.values() if flag_str in str(path)]
        results[multiplier] = [dd.read_parquet(path) for path in result_paths]

    # calculate incidences
    incidences = {}
    for multiplier, simulation_runs in results.items():
        time_series_df = weekly_incidences_from_results(simulation_runs, outcome)
        incidences[multiplier] = time_series_df

    fig, ax = plot_incidences(incidences, 15, title)

    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")
