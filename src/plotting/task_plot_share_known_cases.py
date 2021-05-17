"""Create plots, illustrating the share known cases over time."""
import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import SUMMER_SCENARIO_START
from src.plotting.plotting import plot_share_known_cases
from src.plotting.plotting import PY_DEPENDENCIES
from src.simulation.task_process_simulation_outputs import (
    create_path_for_share_known_cases_of_scenario,
)
from src.simulation.task_process_simulation_outputs import get_available_scenarios
from src.simulation.task_run_simulation import FAST_FLAG
from src.simulation.task_run_simulation import NAMED_SCENARIOS


def create_path_for_share_known_cases_plot(name, fast_flag, groupby):
    if groupby is not None:
        fig_name = f"{fast_flag}_{name}_by_{groupby}.png"
    else:
        fig_name = f"{fast_flag}_{name}.png"
    return BLD / "figures" / "share_known_cases" / fig_name


def create_parametrization(named_scenarios, fast_flag):
    """Create the parametrization for the share known cases plots."""
    available_scenarios = get_available_scenarios(named_scenarios)
    parametrization = []

    for scenario_name in available_scenarios:
        depends_on = create_path_for_share_known_cases_of_scenario(
            scenario_name, fast_flag
        )
        produces = create_path_for_share_known_cases_plot(
            scenario_name, fast_flag, "age_group_rki"
        )
        nice_scenario_name = scenario_name.replace("_", " ").title()
        title = f"Share Known Cases By Age Group in {nice_scenario_name}"
        parametrization.append((depends_on, title, produces))

    return "depends_on, title, produces", parametrization


@pytask.mark.depends_on(PY_DEPENDENCIES)
@pytask.mark.parametrize(*create_parametrization(NAMED_SCENARIOS, FAST_FLAG))
def task_plot_share_known_cases_per_scenario(depends_on, title, produces):
    share_known_cases = pd.read_pickle(depends_on[0])
    fig, ax = plot_share_known_cases(share_known_cases, title)
    if "summer" in title.lower():
        ax.axvline(
            pd.Timestamp(SUMMER_SCENARIO_START),
            label="scenario start",
            color="darkgrey",
        )

    fig.savefig(produces)
    plt.close()
