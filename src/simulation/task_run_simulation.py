import pandas as pd
import pytask
from sid import get_simulate_func

from src.config import FAST_FLAG
from src.simulation.load_params import load_params
from src.simulation.load_simulation_inputs import get_simulation_dependencies
from src.simulation.load_simulation_inputs import load_simulation_inputs
from src.simulation.scenario_config import create_path_to_last_states_of_simulation
from src.simulation.scenario_config import create_path_to_period_outputs_of_simulation
from src.simulation.scenario_config import create_path_to_rapid_test_statistics
from src.simulation.scenario_config import get_named_scenarios


def _create_simulation_parametrization():
    """Convert named scenarios to parametrization.

    Each named scenario is duplicated with different seeds to capture the uncertainty in
    the simulation..

    """
    named_scenarios = get_named_scenarios()

    scenarios = []
    for name, specs in named_scenarios.items():
        is_resumed = specs.get("is_resumed", "fall")
        save_last_states = specs.get("save_last_states", False)
        for seed in range(specs["n_seeds"]):
            produces = {
                "period_outputs": create_path_to_period_outputs_of_simulation(
                    name, seed
                )
            }
            if specs.get("save_rapid_test_statistics", False):
                rapid_test_statistics_path = create_path_to_rapid_test_statistics(
                    name, seed
                )
                produces["rapid_test_statistics"] = rapid_test_statistics_path

                # since we use "append" mode to build this we need to delete the
                # present file with every run
                if rapid_test_statistics_path.exists():
                    rapid_test_statistics_path.unlink()
            else:
                rapid_test_statistics_path = None

            if save_last_states:
                produces["last_states"] = create_path_to_last_states_of_simulation(
                    name, seed
                )

            depends_on = get_simulation_dependencies(
                debug=FAST_FLAG == "debug",
                is_resumed=is_resumed,
            )
            if is_resumed:
                depends_on["initial_states"] = create_path_to_last_states_of_simulation(
                    f"{is_resumed}_baseline", seed
                )

            spec_tuple = (
                depends_on,
                specs["sim_input_scenario"],
                specs["params_scenario"],
                specs["start_date"],
                specs["end_date"],
                save_last_states,
                produces,
                500 + 100_000 * seed,
                is_resumed,
                rapid_test_statistics_path,
            )
            scenarios.append(spec_tuple)

    signature = (
        "depends_on, sim_input_scenario, params_scenario, "
        + "start_date, end_date, save_last_states, produces, seed, "
        + "is_resumed, rapid_test_statistics_path"
    )
    return signature, scenarios


_SIGNATURE, _PARAMETRIZATION = _create_simulation_parametrization()


@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_simulate_scenario(
    depends_on,
    sim_input_scenario,
    params_scenario,
    start_date,
    end_date,
    save_last_states,
    produces,
    seed,
    is_resumed,
    rapid_test_statistics_path,
):
    simulation_kwargs = load_simulation_inputs(
        scenario=sim_input_scenario,
        start_date=start_date,
        end_date=end_date,
        return_last_states=save_last_states,
        debug=FAST_FLAG == "debug",
        period_outputs=True,
        initial_states_path=depends_on["initial_states"],
        is_resumed=is_resumed,
        rapid_test_statistics_path=rapid_test_statistics_path,
    )
    params = load_params(params_scenario)

    simulate = get_simulate_func(
        params=params, path=None, seed=seed, **simulation_kwargs
    )
    res = simulate(params)

    if save_last_states:
        last_states = res.pop("last_states")
        last_states.to_pickle(produces["last_states"])

    pd.to_pickle(res, produces["period_outputs"])
