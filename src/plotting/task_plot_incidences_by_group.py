"""For each available scenario plot the incidences in each of the age groups."""
import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.config import BLD
from src.config import SRC
from src.plotting.plotting import make_nice_outcome
from src.plotting.plotting import OUTCOME_TO_Y_LABEL
from src.plotting.plotting import plot_group_time_series
from src.simulation.load_simulation_inputs import create_period_outputs
from src.simulation.scenario_config import create_path_to_group_incidence_plot
from src.simulation.scenario_config import (
    create_path_to_scenario_outcome_time_series,
)


_DEPENDENCIES = {
    "calculate_moments.py": SRC / "calculate_moments.py",
    "plotting.py": SRC / "plotting" / "plotting.py",
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
    "load_simulation_inputs": SRC / "simulation" / "load_simulation_inputs.py",
}


def create_parametrization():
    entries = [
        entry
        for entry in create_period_outputs()
        if "_by_" in entry and "currently_infected" not in entry
    ]

    parametrization = []
    for entry in entries:
        outcome, groupby = entry.split("_by_")
        depends_on = {
            "simulated": create_path_to_scenario_outcome_time_series(
                scenario_name="combined_baseline", entry=entry
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
            name="combined_baseline", outcome=outcome, groupby=groupby
        )
        parametrization.append((depends_on, produces, outcome, groupby))

    return "depends_on, produces, outcome, groupby", parametrization


_SIGNATURE, _PARAMETRIZATION = create_parametrization()


@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_plot_age_group_incidences_in_one_scenario(
    depends_on, produces, outcome, groupby
):
    incidences = pd.read_pickle(depends_on["simulated"])

    if "rki" in depends_on:
        if groupby == "age_group_rki":
            group_sizes = pd.read_pickle(depends_on["group_sizes_age_groups"])["n"]
        elif groupby == "state":
            state_info = pd.read_parquet(depends_on["group_sizes_states"])
            group_sizes = state_info.set_index("name")["population"]

        rki_data = pd.read_pickle(depends_on["rki"])
        rki_outcome = (
            "newly_infected" if "new_known_case" in produces.name else "newly_deceased"
        )
        rki = (
            smoothed_outcome_per_hundred_thousand_rki(
                df=rki_data,
                outcome=rki_outcome,
                groupby=groupby,
                group_sizes=group_sizes,
                take_logs=False,
            )
            * 7
        )

    else:
        rki = None

    nice_outcome = make_nice_outcome(outcome)
    title = f"{nice_outcome} in " + "{group}"
    ylabel = _get_ylabel(outcome)
    fig, ax = plot_group_time_series(
        df=incidences,
        title=title,
        rki=rki,
        ylabel=ylabel,
    )

    fig.savefig(produces)
    plt.close()


def _get_ylabel(outcome):
    ylabel = OUTCOME_TO_Y_LABEL[outcome]
    if len(ylabel) > 45:
        split = ylabel.split()
        third = int(len(split) / 3)
        ylabel = (
            " ".join(split[:third])
            + "\n"
            + " ".join(split[third : 2 * third])
            + "\n"
            + " ".join(split[2 * third :])
        )

    elif len(ylabel) > 24:
        split = ylabel.split()
        half = int(len(split) / 2)
        ylabel = " ".join(split[:half]) + "\n" + " ".join(split[half:])
    return ylabel
