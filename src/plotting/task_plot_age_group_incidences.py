"""For each available scenario plot the incidences in each of the age groups."""
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns
import sid

from src.plotting.plotting import plot_incidences
from src.plotting.plotting import PY_DEPENDENCIES
from src.simulation.scenario_config import create_path_to_group_incidence_plot
from src.simulation.scenario_config import (
    create_path_to_weekly_outcome_of_scenario,
)
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios


def create_parametrization():
    named_scenarios = get_named_scenarios()
    available_scenarios = get_available_scenarios(named_scenarios)
    outcomes = ["newly_infected", "new_known_case"]
    groupbys = ["state", "age_group_rki"]

    parametrization = []
    for scenario, outcome, groupby in product(available_scenarios, outcomes, groupbys):
        depends_on = create_path_to_weekly_outcome_of_scenario(
            name=scenario, outcome=outcome, groupby=groupby
        )
        produces = create_path_to_group_incidence_plot(
            name=scenario, outcome=outcome, groupby=groupby
        )
        parametrization.append(
            (
                depends_on,
                produces,
            )
        )

    return "depends_on, produces", parametrization


_SIGNATURE, _PARAMETRIZATION = create_parametrization()


@pytask.mark.after_memory_intensive
@pytask.mark.depends_on(PY_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_plot_age_group_incidences_in_one_scenario(depends_on, produces):
    incidences = pd.read_pickle(depends_on[0])
    incidences = incidences.swaplevel()

    group_type = incidences.index.levels[0].name.replace("_", " ")
    title = f"Incidences by {group_type}" + " in {group}"

    fig, ax = _plot_group_incidence(incidences, title)
    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")
    plt.close()


def _plot_group_incidence(incidences, title):
    groups = incidences.index.levels[0].unique()

    if len(groups) < 12:
        colors = sid.get_colors("ordered", len(groups))
    else:
        colors = sns.color_palette("tab20b")[: len(groups)]

    n_rows = int(np.ceil(len(groups) / 2))
    fig, axes = plt.subplots(figsize=(12, n_rows * 3), nrows=n_rows, ncols=2)
    axes = axes.flatten()
    for group, ax in zip(groups, axes):
        plot_incidences(
            incidences={group: incidences.loc[group]},
            title=title.format(group=group),
            colors=colors,
            name_to_label={},
            rki=False,
            plot_scenario_start=False,
            fig=fig,
            ax=ax,
        )
    return fig, ax
