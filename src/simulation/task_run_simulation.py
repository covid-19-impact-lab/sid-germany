import pytask
from sid import get_simulate_func

from src.config import FAST_FLAG
from src.simulation.load_params import load_params
from src.simulation.load_simulation_inputs import get_simulation_dependencies
from src.simulation.load_simulation_inputs import load_simulation_inputs
from src.simulation.load_simulation_inputs import named_scenarios_to_parametrization


DEPENDENCIES = get_simulation_dependencies(debug=FAST_FLAG == "debug")


if FAST_FLAG == "debug":
    n_baseline_seeds = 1
    n_scenario_seeds = 0
elif FAST_FLAG == "verify":
    n_baseline_seeds = 18
    n_scenario_seeds = 5
elif FAST_FLAG == "full":
    n_baseline_seeds = 20
    n_scenario_seeds = 20
else:
    raise ValueError(
        f"Unknown FAST_FLAG {FAST_FLAG}."
        "Only 'debug', 'verify' or 'full' are allowed."
    )


NAMED_SCENARIOS = {
    "fall_baseline": {
        "policy_scenario": "baseline",
        "params_scenario": "baseline",
        "start_date": "2020-10-01",
        "end_date": "2020-12-23",
        "n_seeds": n_baseline_seeds,
    },
    "spring_baseline": {
        "policy_scenario": "baseline",
        "params_scenario": "baseline",
        "start_date": "2021-02-15",
        "end_date": "2021-05-07",
        "n_seeds": n_baseline_seeds,
    },
}


SCENARIOS = named_scenarios_to_parametrization(NAMED_SCENARIOS, FAST_FLAG)


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.parametrize(
    "policy_scenario, params_scenario, start_date, end_date, produces, seed",
    SCENARIOS,
)
def task_simulate_scenario(
    policy_scenario,
    params_scenario,
    start_date,
    end_date,
    produces,
    seed,
):
    simulation_kwargs = load_simulation_inputs(
        scenario=policy_scenario,
        start_date=start_date,
        end_date=end_date,
        debug=FAST_FLAG == "debug",
    )
    params = load_params(params_scenario)
    path = produces.parent.parent

    simulate = get_simulate_func(
        params=params, path=path, seed=seed, **simulation_kwargs
    )
    simulate(params)
