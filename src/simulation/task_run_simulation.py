import pandas as pd
import pytask
from sid import get_simulate_func

from src.config import FAST_FLAG
from src.policies.policy_tools import combine_dictionaries
from src.simulation.load_params import load_params
from src.simulation.load_simulation_inputs import get_simulation_dependencies
from src.simulation.load_simulation_inputs import load_simulation_inputs
from src.simulation.scenario_config import (
    create_path_to_initial_group_share_known_cases,
)
from src.simulation.scenario_config import create_path_to_period_outputs_of_simulation
from src.simulation.scenario_config import get_named_scenarios


def _create_simulation_parametrization():
    """Convert named scenarios to parametrization.

    Each named scenario is duplicated with different seeds to capture the uncertainty in
    the simulation..

    """
    named_scenarios = get_named_scenarios()
    common_dependencies = get_simulation_dependencies(debug=FAST_FLAG == "debug")
    fall_end_date = pd.Timestamp(named_scenarios["fall_baseline"]["end_date"])

    scenarios = []
    for name, specs in named_scenarios.items():
        if pd.Timestamp(specs["start_date"]) > pd.Timestamp("2020-11-01"):
            skc_path = create_path_to_initial_group_share_known_cases(
                "fall_baseline", fall_end_date
            )
            group_share_dependencies = {"group_share_known_case_path": skc_path}
        else:
            group_share_dependencies = {}
        for seed in range(specs["n_seeds"]):
            produces = create_path_to_period_outputs_of_simulation(name, seed)
            scaled_seed = 500 + 100_000 * seed
            depends_on = combine_dictionaries(
                [common_dependencies, group_share_dependencies]
            )
            spec_tuple = (
                depends_on,
                specs["sim_input_scenario"],
                specs["params_scenario"],
                specs["start_date"],
                specs["end_date"],
                produces,
                scaled_seed,
            )
            scenarios.append(spec_tuple)

    signature = (
        "depends_on, sim_input_scenario, params_scenario, "
        + "start_date, end_date, produces, seed"
    )
    return signature, scenarios


_SIGNATURE, _PARAMETRIZATION = _create_simulation_parametrization()


@pytask.mark.skip
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_simulate_scenario(
    depends_on,
    sim_input_scenario,
    params_scenario,
    start_date,
    end_date,
    produces,
    seed,
):
    group_share_known_case_path = depends_on.get("group_share_known_case_path")
    simulation_kwargs = load_simulation_inputs(
        scenario=sim_input_scenario,
        start_date=start_date,
        end_date=end_date,
        debug=FAST_FLAG == "debug",
        group_share_known_case_path=group_share_known_case_path,
        period_outputs=True,
    )
    params = load_params(params_scenario)

    simulate = get_simulate_func(
        params=params, path=None, seed=seed, **simulation_kwargs
    )
    res = simulate(params)
    pd.to_pickle(res, produces)
