"""For each available scenario plot the incidences in each of the age groups."""
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns
import sid

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.config import BLD
from src.config import SRC
from src.plotting.plotting import plot_incidences
from src.simulation.scenario_config import create_path_to_group_incidence_plot
from src.simulation.scenario_config import (
    create_path_to_weekly_outcome_of_scenario,
)
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios

_DEPENDENCIES = {
    "calculate_moments": SRC / "calculate_moments.py",
    "config": SRC / "config.py",
    "plotting": SRC / "plotting" / "plotting.py",
    "scenario_config": SRC / "simulation" / "scenario_config.py",
}


def create_parametrization():
    named_scenarios = get_named_scenarios()
    available_scenarios = get_available_scenarios(named_scenarios)
    outcomes = ["newly_infected", "new_known_case"]
    groupbys = ["state", "age_group_rki"]

    parametrization = []
    for scenario, outcome, groupby in product(available_scenarios, outcomes, groupbys):
        depends_on = {
            "simulated": create_path_to_weekly_outcome_of_scenario(
                name=scenario, entry=f"{outcome}_by_{groupby}"
            ),
            "group_sizes_age_groups": (
                BLD / "data" / "population_structure" / "age_groups_rki.pkl"
            ),
            "group_sizes_states": (
                BLD / "data" / "population_structure" / "federal_states.parquet"
            ),
        }
        if outcome == "new_known_case":
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
    incidences = incidences.swaplevel()

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
    fig, ax = _plot_group_incidence(incidences, title, rki)

    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")
    plt.close()


def _plot_group_incidence(incidences, title, rki):
    groups = incidences.index.levels[0].unique()
    dates = incidences.index.levels[1].unique()

    n_rows = int(np.ceil(len(groups) / 2))
    fig, axes = plt.subplots(figsize=(12, n_rows * 3), nrows=n_rows, ncols=2)
    axes = axes.flatten()
    sid_blue = sid.get_colors("categorical", 5)[0]
    for group, ax in zip(groups, axes):
        plot_incidences(
            incidences={group: incidences.loc[group]},
            title=title.format(group=group),
            colors=[sid_blue],
            name_to_label={group: "simulated"},
            rki=False,
            plot_scenario_start=False,
            fig=fig,
            ax=ax,
        )

        if rki is not None:
            rki_data = rki.loc[dates, group].reset_index()
            sns.lineplot(
                x=rki_data["date"],
                y=rki_data[0],
                ax=ax,
                color="k",
                label="official case numbers",
            )

    return fig, ax
