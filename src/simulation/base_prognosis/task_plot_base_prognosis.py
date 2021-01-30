import dask.dataframe as dd
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.simulation.base_prognosis_specification import (
    build_base_prognosis_parametrization,
)
from src.simulation.plotting import plot_incidences
from src.simulation.plotting import weekly_incidences_from_results

NESTED_PARAMETRIZATION = build_base_prognosis_parametrization()

DEPENDENCIES = {
    "specs": SRC / "simulation" / "base_prognosis_specification.py",
}
for name, seed_list in NESTED_PARAMETRIZATION.items():
    for path, _, seed in seed_list:
        DEPENDENCIES[(name, seed)] = path

INCIDENCE_PATHS = {
    "all": BLD / "simulations" / "base_prognosis" / "all_incidences.pkl",
    "means": BLD / "simulations" / "base_prognosis" / "mean_incidences.csv",
}


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.produces(INCIDENCE_PATHS)
def task_save_results(depends_on, produces):
    depends_on.pop("specs")
    results = {}
    scenario_names = NESTED_PARAMETRIZATION.keys()
    for name in scenario_names:
        paths = [path for (scenario, _), path in depends_on.items() if name in scenario]
        results[name] = [dd.read_parquet(p) for p in paths]

    incidences = {}
    for outcome in ["new_known_case", "newly_infected"]:
        for name, simulation_runs in results.items():
            time_series_df = weekly_incidences_from_results(simulation_runs, outcome)
            incidences[(name, outcome)] = time_series_df

    pd.to_pickle(incidences, produces["all"])

    to_concat = [
        pd.Series(df.mean(axis=1), name=name) for name, df in incidences.items()
    ]
    means = pd.concat(to_concat, axis=1)
    means.to_csv(produces["means"])


PLOT_DEPENDENCIES = {
    "plotting_py": SRC / "simulation" / "plotting.py",
    "rki_data": BLD / "data" / "processed_time_series" / "rki.pkl",
    **INCIDENCE_PATHS,
}

PLOT_PARAMETRIZATION = []
for outcome, title in [
    ("new_known_case", "Beobachtete Inzidenz"),
    ("newly_infected", "Tatsächliche Inzidenz"),
]:
    produces = {"fig": BLD / "simulations" / "base_prognosis" / f"{outcome}.png"}
    spec = (outcome, title, produces)
    PLOT_PARAMETRIZATION.append(spec)


@pytask.mark.depends_on(PLOT_DEPENDENCIES)
@pytask.mark.parametrize("outcome, title, produces", PLOT_PARAMETRIZATION)
def task_plot_base_prognosis(depends_on, outcome, title, produces):
    incidences = pd.read_pickle(depends_on["all"])
    to_plot = {key[0]: df for key, df in incidences.items() if key[1] == outcome}

    fig, ax = plot_incidences(to_plot, 15, title, rki=outcome)
    fig.savefig(produces["fig"], dpi=200, transparent=False, facecolor="w")
