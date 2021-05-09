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


def create_path_for_figure_of_weekly_outcome_of_scenario(name, fast_flag, outcome):
    return BLD / "figures" / f"{fast_flag}_{name}_{outcome}.png"


def create_parametrization(named_scenarios, fast_flag, outcomes):
    parametrization = []
    for outcome in outcomes:
        for name in named_scenarios:
            depends_on = create_path_for_weekly_outcome_of_scenario(
                name, fast_flag, outcome
            )
            produces = create_path_for_figure_of_weekly_outcome_of_scenario(
                name, fast_flag, outcome
            )
            parametrization.append((depends_on, name, outcome, produces))

    return "depends_on, name, outcome, produces", parametrization


SIGNATURE, PARAMETRIZATION = create_parametrization(
    NAMED_SCENARIOS, FAST_FLAG, OUTCOMES
)


@pytask.mark.parametrize(SIGNATURE, PARAMETRIZATION)
def task_plot_weekly_outcomes(depends_on, name, outcome, produces):
    df = pd.read_parquet(depends_on)
    fig, ax = plot_incidences(
        {name: df}, df.shape[1], outcome, {name: name}, rki=outcome
    )
    plt.savefig(produces)
