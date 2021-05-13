import dask.dataframe as dd
import pytask

from src.config import BLD
from src.plotting.plotting import weekly_incidences_from_results
from src.simulation.load_simulation_inputs import (
    create_path_to_last_states_of_simulation,
)
from src.simulation.task_run_simulation import FAST_FLAG
from src.simulation.task_run_simulation import NAMED_SCENARIOS


OUTCOMES = [
    "newly_infected",
    "new_known_case",
    "currently_infected",
    "knows_currently_infected",
]


def create_path_for_weekly_outcome_of_scenario(name, fast_flag, outcome, groupby):
    if groupby is None:
        file_name = f"{fast_flag}_{name}_{outcome}.pickle"
    else:
        file_name = f"{fast_flag}_{name}_{outcome}_by_{groupby}.pickle"
    return BLD / "simulations" / file_name


def create_parametrization(named_scenarios, fast_flag, outcomes):
    parametrization = []
    for outcome in outcomes:
        for groupby in [None, "age_group_rki"]:
            for name, specs in named_scenarios.items():
                dependencies = {
                    seed: create_path_to_last_states_of_simulation(
                        fast_flag, name, seed
                    )
                    for seed in range(specs["n_seeds"])
                }

                # this handles the case of 0 seeds, i.e. skipped scenarios
                if dependencies:
                    produces = create_path_for_weekly_outcome_of_scenario(
                        name, fast_flag, outcome, groupby
                    )

                    parametrization.append((dependencies, outcome, groupby, produces))

    return "depends_on, outcome, groupby, produces", parametrization


SIGNATURE, PARAMETRIZATION = create_parametrization(
    NAMED_SCENARIOS, FAST_FLAG, OUTCOMES
)


@pytask.mark.parametrize(SIGNATURE, PARAMETRIZATION)
def task_create_weekly_outcome_for_scenario(depends_on, outcome, groupby, produces):
    ddfs = {
        seed: dd.read_parquet(depends_on[seed].parent.parent.joinpath("time_series"))
        for seed in sorted(depends_on)
    }
    weekly_outcomes = weekly_incidences_from_results(ddfs.values(), outcome, groupby)
    weekly_outcomes = weekly_outcomes.rename(columns=str)
    weekly_outcomes.to_pickle(produces)
