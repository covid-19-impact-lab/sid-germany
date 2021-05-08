from pathlib import Path

import pytask
from sid import get_simulate_func

from src.config import BLD
from src.config import FAST_FLAG
from src.simulation.load_params import load_params
from src.simulation.load_simulation_inputs import get_simulation_dependencies
from src.simulation.load_simulation_inputs import load_simulation_inputs

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

spring_dates = {
    "start_date": "2021-02-15",
    "end_date": "2021-05-07",
}

NAMED_SCENARIOS = {
    "fall_baseline": {
        "sim_input_scenario": "baseline",
        "params_scenario": "baseline",
        "start_date": "2020-10-01",
        "end_date": "2020-12-23",
        "n_seeds": n_baseline_seeds,
    },
    "spring_baseline": {
        "sim_input_scenario": "baseline",
        "params_scenario": "baseline",
        "n_seeds": n_baseline_seeds,
        **spring_dates,
    },
    "spring_without_vaccines": {
        "sim_input_scenario": "no_vaccinations_after_feb_15",
        "params_scenario": "baseline",
        "n_seeds": n_baseline_seeds,
        **spring_dates,
    },
    "spring_without_rapid_tests_at_schools": {
        "sim_input_scenario": "baseline",
        "params_scenario": "no_rapid_tests_at_schools",
        "n_seeds": n_baseline_seeds,
        **spring_dates,
    },
    "spring_without_rapid_tests_at_work": {
        "sim_input_scenario": "baseline",
        "params_scenario": "no_rapid_tests_at_work",
        "n_seeds": n_baseline_seeds,
        **spring_dates,
    },
    "spring_without_rapid_tests": {
        "sim_input_scenario": "no_rapid_tests",
        "params_scenario": "baseline",
        "n_seeds": n_baseline_seeds,
        **spring_dates,
    },
    "spring_emergency_care_after_easter_no_school_rapid_tests": {
        "sim_input_scenario": "only_strict_emergency_care_after_april_5",
        "params_scenario": "no_rapid_tests_at_schools",
        "n_seeds": n_baseline_seeds,
        **spring_dates,
    },
}

SCENARIOS = []
for name, specs in NAMED_SCENARIOS.items():
    for seed in range(specs["n_seeds"]):
        produces = Path(
            BLD
            / "simulations"
            / f"{FAST_FLAG}_{name}_{seed}"
            / "last_states"
            / "last_states.parquet"
        )
        scaled_seed = 500 + 100_000 * seed
        spec_tuple = (
            specs["sim_input_scenario"],
            specs["params_scenario"],
            specs["start_date"],
            specs["end_date"],
            produces,
            scaled_seed,
        )
        SCENARIOS.append(spec_tuple)


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.parametrize(
    "sim_input_scenario, params_scenario, start_date, end_date, produces, seed",
    SCENARIOS,
)
def task_simulate_scenario(
    sim_input_scenario,
    params_scenario,
    start_date,
    end_date,
    produces,
    seed,
):
    simulation_kwargs = load_simulation_inputs(
        scenario=sim_input_scenario,
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
