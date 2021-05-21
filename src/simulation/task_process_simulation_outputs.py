import pandas as pd
import pytask

from src.calculate_moments import aggregate_and_smooth_period_outcome_sim
from src.config import SRC
from src.simulation.load_simulation_inputs import create_period_outputs
from src.simulation.scenario_config import create_path_to_period_outputs_of_simulation
from src.simulation.scenario_config import create_path_to_share_known_cases_of_scenario
from src.simulation.scenario_config import create_path_to_weekly_outcome_of_scenario
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios


def _create_create_weekly_incidence_parametrization():
    named_scenarios = get_named_scenarios()
    parametrization = []
    period_output_keys = create_period_outputs().keys()
    for name, specs in named_scenarios.items():
        dependencies = {
            seed: create_path_to_period_outputs_of_simulation(name, seed)
            for seed in range(specs["n_seeds"])
        }
        # this handles the case of 0 seeds, i.e. skipped scenarios
        if dependencies:
            produces = {
                entry: create_path_to_weekly_outcome_of_scenario(name, entry)
                for entry in period_output_keys
            }
            parametrization.append((dependencies, produces))

    return "depends_on, produces", parametrization


_SIGNATURE, _PARAMETRIZATION = _create_create_weekly_incidence_parametrization()


@pytask.mark.depends_on({"calculate_weekly_incidences": SRC / "calculate_moments.py"})
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_create_weekly_outcome_for_scenario(depends_on, produces):
    seed_keys = [seed for seed in depends_on if isinstance(seed, int)]
    results = {str(seed): pd.read_pickle(depends_on[seed]) for seed in seed_keys}
    for entry, path in produces.items():
        outcome_and_groupby = entry.split("_by_")
        weekly_outcomes = pd.DataFrame()
        groupby = None if len(outcome_and_groupby) == 1 else outcome_and_groupby[1]
        for seed, res in results.items():
            weekly_outcomes[seed] = aggregate_and_smooth_period_outcome_sim(
                res, outcome=entry, groupby=groupby, take_logs=False
            )
        weekly_outcomes.to_pickle(path)


def _create_scenario_share_known_cases_parametrization():
    """Create the parametrization for the share known cases."""
    named_scenarios = get_named_scenarios()
    available_scenarios = get_available_scenarios(named_scenarios)
    parametrization = []

    for scenario_name in available_scenarios:
        depends_on = {
            outcome: create_path_to_weekly_outcome_of_scenario(
                scenario_name, f"{outcome}_by_age_group_rki"
            )
            for outcome in ["currently_infected", "knows_currently_infected"]
        }
        produces = create_path_to_share_known_cases_of_scenario(scenario_name)
        parametrization.append((depends_on, produces))
    return "depends_on, produces", parametrization


_SIGNATURE, _PARAMETRIZATION = _create_scenario_share_known_cases_parametrization()


@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_create_share_known_cases(depends_on, produces):
    knows_currently_infected = pd.read_pickle(depends_on["knows_currently_infected"])
    currently_infected = pd.read_pickle(depends_on["currently_infected"])
    share_known_cases = knows_currently_infected / currently_infected
    share_known_cases["mean"] = share_known_cases.mean(axis=1)
    assert not share_known_cases.index.duplicated().any()
    assert share_known_cases.notnull().all().all()
    share_known_cases.to_pickle(produces)
