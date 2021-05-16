"""Create plots, illustrating the share known cases over time."""
import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.config import SUMMER_SCENARIO_START
from src.plotting.plotting import plot_share_known_cases
from src.simulation.task_process_simulation_outputs import (
    create_path_for_weekly_outcome_of_scenario,
)
from src.simulation.task_run_simulation import FAST_FLAG
from src.simulation.task_run_simulation import NAMED_SCENARIOS


PY_DEPENDENCIES = {
    "py_config": SRC / "config.py",
    "py_plot_incidences": SRC / "plotting" / "plotting.py",
    "py_process_sim_outputs": SRC / "simulation" / "task_process_simulation_outputs.py",
}


def create_path_for_share_known_cases_plot(name, fast_flag, groupby):
    if groupby is not None:
        fig_name = f"{fast_flag}_{name}_by_{groupby}.png"
    else:
        fig_name = f"{fast_flag}_{name}.png"
    return BLD / "figures" / "share_known_cases" / fig_name


def create_parametrization(named_scenarios, fast_flag):
    """Create the parametrization for the share known cases plots."""
    available_scenarios = {
        name for name, spec in named_scenarios.items() if spec["n_seeds"] > 0
    }
    parametrization = []
    outcomes = ["currently_infected", "knows_currently_infected"]

    # Age Group Plots
    for scenario_name in available_scenarios:
        depends_on = {
            outcome: create_path_for_weekly_outcome_of_scenario(
                scenario_name, fast_flag, outcome, "age_group_rki"
            )
            for outcome in outcomes
        }
        produces = create_path_for_share_known_cases_plot(
            scenario_name, FAST_FLAG, "age_group_rki"
        )
        nice_scenario_name = scenario_name.replace("_", " ").title()
        title = f"Share Known Cases By Age Group in {nice_scenario_name}"
        parametrization.append((depends_on, title, produces))

    return "depends_on, title, produces", parametrization


SIGNATURE, PARAMETRIZATION = create_parametrization(NAMED_SCENARIOS, FAST_FLAG)


@pytask.mark.depends_on(PY_DEPENDENCIES)
@pytask.mark.parametrize(SIGNATURE, PARAMETRIZATION)
def task_plot_share_known_cases_per_scenario(depends_on, title, produces):
    knows_currently_infected = pd.read_pickle(depends_on["knows_currently_infected"])
    currently_infected = pd.read_pickle(depends_on["currently_infected"])
    share_known_cases = knows_currently_infected / currently_infected
    share_known_cases["mean"] = share_known_cases.mean(axis=1)
    fig, ax = plot_share_known_cases(share_known_cases, title)
    if "summer" in title.lower():
        ax.axvline(
            pd.Timestamp(SUMMER_SCENARIO_START),
            label="scenario start",
            color="darkgrey",
        )

    fig.savefig(produces)
    plt.close()
