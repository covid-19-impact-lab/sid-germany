import dask.dataframe as dd
import pandas as pd
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
        file_name = f"{fast_flag}_{name}_{outcome}.pkl"
    else:
        file_name = f"{fast_flag}_{name}_{outcome}_by_{groupby}.pkl"
    return BLD / "simulations" / file_name


def create_incidence_parametrization(named_scenarios, fast_flag, outcomes):
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


SIGNATURE, PARAMETRIZATION = create_incidence_parametrization(
    NAMED_SCENARIOS, FAST_FLAG, OUTCOMES
)


@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_create_weekly_outcome_for_scenario(depends_on, outcome, groupby, produces):
    ddfs = {
        seed: dd.read_parquet(depends_on[seed].parent.parent.joinpath("time_series"))
        for seed in sorted(depends_on)
    }
    weekly_outcomes = weekly_incidences_from_results(ddfs.values(), outcome, groupby)
    weekly_outcomes = weekly_outcomes.rename(columns=str)
    weekly_outcomes.to_pickle(produces)


def create_path_for_share_known_cases_of_scenario(name, fast_flag):
    return BLD / "simulations" / f"{fast_flag}_{name}_share_known_cases.pkl"


def create_share_known_cases_parametrization(named_scenarios, fast_flag):
    """Create the parametrization for the share known cases."""
    available_scenarios = {
        name for name, spec in named_scenarios.items() if spec["n_seeds"] > 0
    }
    parametrization = []
    outcomes = ["currently_infected", "knows_currently_infected"]

    for scenario_name in available_scenarios:
        depends_on = {
            outcome: create_path_for_weekly_outcome_of_scenario(
                scenario_name, fast_flag, outcome, "age_group_rki"
            )
            for outcome in outcomes
        }
        produces = create_path_for_share_known_cases_of_scenario(
            scenario_name, fast_flag
        )
        parametrization.append((depends_on, produces))
    return parametrization


SHARE_KNOWN_CASES_PARAMETRIZATION = create_share_known_cases_parametrization(
    NAMED_SCENARIOS, FAST_FLAG
)


@pytask.mark.parametrize("depends_on, produces", SHARE_KNOWN_CASES_PARAMETRIZATION)
def task_create_share_known_cases(depends_on, produces):
    knows_currently_infected = pd.read_pickle(depends_on["knows_currently_infected"])
    currently_infected = pd.read_pickle(depends_on["currently_infected"])
    share_known_cases = knows_currently_infected / currently_infected
    share_known_cases["mean"] = share_known_cases.mean(axis=1)
    share_known_cases.to_pickle(produces)
