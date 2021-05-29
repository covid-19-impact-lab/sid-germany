"""For each available scenario plot the incidences in each of the age groups."""
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.config import BLD
from src.config import SRC
from src.plotting.plotting import plot_group_time_series
from src.simulation.load_simulation_inputs import create_period_outputs
from src.simulation.scenario_config import create_path_to_group_incidence_plot
from src.simulation.scenario_config import (
    create_path_to_scenario_outcome_time_series,
)
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios


_DEPENDENCIES = {
    "calculate_moments.py": SRC / "calculate_moments.py",
    "plotting.py": SRC / "plotting" / "plotting.py",
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
    "outputs": SRC / "plotting" / "task_plot_scenario_comparisons.py",
}


def create_parametrization():
    named_scenarios = get_named_scenarios()
    available_scenarios = get_available_scenarios(named_scenarios)
    entries = [entry for entry in create_period_outputs().keys() if "_by_" in entry]

    parametrization = []
    for scenario, entry in product(available_scenarios, entries):
        outcome, groupby = entry.split("_by_")
        depends_on = {
            "simulated": create_path_to_scenario_outcome_time_series(
                scenario_name=scenario, entry=entry
            ),
            "group_sizes_age_groups": (
                BLD / "data" / "population_structure" / "age_groups_rki.pkl"
            ),
            "group_sizes_states": (
                BLD / "data" / "population_structure" / "federal_states.parquet"
            ),
        }
        if outcome in ["new_known_case", "newly_deceased"]:
            depends_on["rki"] = BLD / "data" / "processed_time_series" / "rki.pkl"

        produces = create_path_to_group_incidence_plot(
            name=scenario, outcome=outcome, groupby=groupby
        )
        parametrization.append((depends_on, produces, groupby))

    return "depends_on, produces, groupby", parametrization


_SIGNATURE, _PARAMETRIZATION = create_parametrization()


@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_plot_age_group_incidences_in_one_scenario(depends_on, produces, groupby):
    incidences = pd.read_pickle(depends_on["simulated"])

    if "rki" in depends_on:
        if groupby == "age_group_rki":
            group_sizes = pd.read_pickle(depends_on["group_sizes_age_groups"])["n"]
        elif groupby == "state":
            state_info = pd.read_parquet(depends_on["group_sizes_states"])
            group_sizes = state_info.set_index("name")["population"]

        rki_data = pd.read_pickle(depends_on["rki"])
        rki = (
            smoothed_outcome_per_hundred_thousand_rki(
                df=rki_data,
                outcome="newly_infected",
                groupby=groupby,
                group_sizes=group_sizes,
                take_logs=False,
            )
            * 7
        )

    else:
        rki = None

    title = f"Incidences by {groupby.replace('_', ' ')} in " + "{group}"
    fig, ax = plot_group_time_series(incidences, title, rki)

    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")
    plt.close()
