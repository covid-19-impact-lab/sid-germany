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
    n_main_scenario_seeds = 0
    n_side_scenario_seeds = 0
elif FAST_FLAG == "verify":
    n_baseline_seeds = 18
    n_main_scenario_seeds = 5
    n_side_scenario_seeds = 0
elif FAST_FLAG == "full":
    n_baseline_seeds = 20
    n_main_scenario_seeds = 20
    n_side_scenario_seeds = 20
else:
    raise ValueError(
        f"Unknown FAST_FLAG {FAST_FLAG}."
        "Only 'debug', 'verify' or 'full' are allowed."
    )

spring_dates = {
    "start_date": "2021-02-15",
    "end_date": "2021-05-07",
}

prediction_dates = {
    "start_date": "2021-04-15",
    "end_date": "2021-07-01",
}

NAMED_SCENARIOS = {
    # Baseline Scenarios
    "fall_baseline": {
        "sim_input_scenario": "baseline",
        "params_scenario": "baseline",
        "start_date": "2020-10-15",
        "end_date": "2020-12-23",
        "n_seeds": n_baseline_seeds,
    },
    "spring_baseline": {
        "sim_input_scenario": "baseline",
        "params_scenario": "baseline",
        "n_seeds": n_baseline_seeds,
        **spring_dates,
    },
    "future_baseline": {
        "sim_input_scenario": "baseline",
        "params_scenario": "baseline",
        "n_seeds": n_baseline_seeds,
        **prediction_dates,
    },
    # Policy Scenarios
    "spring_without_vaccines": {
        "sim_input_scenario": "no_vaccinations_after_feb_15",
        "params_scenario": "baseline",
        "n_seeds": n_main_scenario_seeds,
        **spring_dates,
    },
    "spring_without_rapid_tests_at_schools": {
        "sim_input_scenario": "baseline",
        "params_scenario": "no_rapid_tests_at_schools",
        "n_seeds": n_side_scenario_seeds,
        **spring_dates,
    },
    "spring_without_rapid_tests_at_work": {
        "sim_input_scenario": "baseline",
        "params_scenario": "no_rapid_tests_at_work",
        "n_seeds": n_side_scenario_seeds,
        **spring_dates,
    },
    "spring_without_rapid_tests": {
        "sim_input_scenario": "no_rapid_tests",
        "params_scenario": "baseline",
        "n_seeds": n_side_scenario_seeds,
        **spring_dates,
    },
    # Rapid Tests vs School Closures
    "spring_emergency_care_after_easter_no_school_rapid_tests": {
        "sim_input_scenario": "only_strict_emergency_care_after_april_5",
        "params_scenario": "no_rapid_tests_at_schools",
        "n_seeds": n_side_scenario_seeds,
        **spring_dates,
    },
    "spring_educ_open_after_easter": {
        "sim_input_scenario": "open_all_educ_after_easter",
        "params_scenario": "baseline",
        "n_seeds": n_main_scenario_seeds,
        **spring_dates,
    },
    "spring_educ_open_after_easter_educ_tests_every_other_day": {
        "sim_input_scenario": "open_all_educ_after_easter",
        "params_scenario": "rapid_tests_at_school_every_other_day_after_april_5",
        "n_seeds": n_side_scenario_seeds,
        **spring_dates,
    },
    "spring_educ_open_after_easter_educ_tests_every_day": {
        "sim_input_scenario": "open_all_educ_after_easter",
        "params_scenario": "rapid_tests_at_school_every_day_after_april_5",
        "n_seeds": n_side_scenario_seeds,
        **spring_dates,
    },
    # Future Scenarios
    "future_educ_open": {
        "sim_input_scenario": "open_all_educ_after_scenario_start",
        "params_scenario": "baseline",
        "n_seeds": n_main_scenario_seeds,
        **prediction_dates,
    },
    "future_reduced_test_demand": {
        "sim_input_scenario": "baseline",
        "params_scenario": "reduce_rapid_test_demand_after_may_17",
        "n_seeds": n_main_scenario_seeds,
        **prediction_dates,
    },
}

SCENARIOS = named_scenarios_to_parametrization(NAMED_SCENARIOS, FAST_FLAG)


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
