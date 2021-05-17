import dask.dataframe as dd
import pandas as pd
import pytask

from src.calculate_moments import calculate_weekly_incidences_from_results
from src.config import SRC
from src.simulation.shared import create_path_to_last_states_of_simulation
from src.simulation.shared import create_path_to_share_known_cases_of_scenario
from src.simulation.shared import create_path_to_weekly_outcome_of_scenario
from src.simulation.shared import get_available_scenarios
from src.simulation.shared import get_named_scenarios

OUTCOMES = [
    "newly_infected",
    "new_known_case",
    "currently_infected",
    "knows_currently_infected",
]


def _create_create_weekly_incidence_parametrization(outcomes):
    named_scenarios = get_named_scenarios()
    parametrization = []
    for outcome in outcomes:
        for groupby in [None, "age_group_rki"]:
            for name, specs in named_scenarios.items():
                dependencies = {
                    seed: create_path_to_last_states_of_simulation(name, seed)
                    for seed in range(specs["n_seeds"])
                }

                # this handles the case of 0 seeds, i.e. skipped scenarios
                if dependencies:
                    produces = create_path_to_weekly_outcome_of_scenario(
                        name, outcome, groupby
                    )

                    parametrization.append((dependencies, outcome, groupby, produces))

    return "depends_on, outcome, groupby, produces", parametrization


def _create_calculate_share_known_cases_of_scenarios_parametrization():
    """Create the parametrization for the share known cases."""
    named_scenarios = get_named_scenarios()
    available_scenarios = get_available_scenarios(named_scenarios)
    parametrization = []
    outcomes = ["currently_infected", "knows_currently_infected"]

    for scenario_name in available_scenarios:
        depends_on = {
            outcome: create_path_to_weekly_outcome_of_scenario(
                scenario_name, outcome, "age_group_rki"
            )
            for outcome in outcomes
        }
        produces = create_path_to_share_known_cases_of_scenario(scenario_name)
        parametrization.append((depends_on, produces))
    return "depends_on, produces", parametrization


@pytask.mark.depends_on({"calculate_weekly_incidences": SRC / "calculate_moments.py"})
@pytask.mark.parametrize(*_create_create_weekly_incidence_parametrization(OUTCOMES))
def task_create_weekly_outcome_for_scenario(depends_on, outcome, groupby, produces):
    ddfs = {
        seed: dd.read_parquet(depends_on[seed].parent.parent.joinpath("time_series"))
        for seed in depends_on
        if isinstance(seed, int)
    }
    weekly_outcomes = calculate_weekly_incidences_from_results(
        ddfs.values(), outcome, groupby
    )
    weekly_outcomes = weekly_outcomes.rename(columns=str)
    weekly_outcomes.to_pickle(produces)


@pytask.mark.parametrize(
    *_create_calculate_share_known_cases_of_scenarios_parametrization()
)
def task_create_share_known_cases(depends_on, produces):
    knows_currently_infected = pd.read_pickle(depends_on["knows_currently_infected"])
    currently_infected = pd.read_pickle(depends_on["currently_infected"])
    share_known_cases = knows_currently_infected / currently_infected
    share_known_cases["mean"] = share_known_cases.mean(axis=1)
    assert not share_known_cases.index.duplicated().any()
    share_known_cases.to_pickle(produces)
