import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.plotting.plotting import plot_incidences
from src.simulation.task_process_simulation_outputs import (
    create_path_for_weekly_outcome_of_scenario,
)
from src.simulation.task_process_simulation_outputs import OUTCOMES
from src.simulation.task_run_simulation import FAST_FLAG
from src.simulation.task_run_simulation import NAMED_SCENARIOS


PLOTS = {
    "fall": ["fall_baseline"],
    "spring": ["spring_baseline"],
}
"""Dict[str, List[str]]: A dictionary containing the plots to create.

Each key in the dictionary is a name for a collection of scenarios. The values are lists
of scenario names which are combined to create the collection.

"""


def create_path_for_figure_of_weekly_outcome_of_scenario(name, fast_flag, outcome):
    return BLD / "figures" / f"{fast_flag}_{name}_{outcome}.png"


def create_parametrization(plots, named_scenarios, fast_flag, outcomes):
    parametrization = []
    for outcome in outcomes:
        for plot_name, plot in plots.items():
            depends_on = {
                scenario_name: create_path_for_weekly_outcome_of_scenario(
                    scenario_name, fast_flag, outcome
                )
                for scenario_name in plot
            }

            missing_scenarios = set(depends_on) - set(named_scenarios)
            if missing_scenarios:
                raise ValueError(f"Some scenarios are missing: {missing_scenarios}.")

            produces = create_path_for_figure_of_weekly_outcome_of_scenario(
                plot_name, fast_flag, outcome
            )
            parametrization.append((depends_on, plot_name, outcome, produces))

    return "depends_on, plot_name, outcome, produces", parametrization


SIGNATURE, PARAMETRIZATION = create_parametrization(
    PLOTS, NAMED_SCENARIOS, FAST_FLAG, OUTCOMES
)


@pytask.mark.parametrize(SIGNATURE, PARAMETRIZATION)
def task_plot_weekly_outcomes(depends_on, plot_name, outcome, produces):
    dfs = {name: pd.read_parquet(path) for name, path in depends_on.items()}
    fig, ax = plot_incidences(
        dfs, plot_name, {name: name for name in dfs}, rki=outcome == "new_known_case"
    )
    plt.savefig(produces)
