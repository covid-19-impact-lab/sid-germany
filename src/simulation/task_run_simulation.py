import pytask
from sid import get_simulate_func

from src.config import FAST_FLAG
from src.simulation.load_params import load_params
from src.simulation.load_simulation_inputs import (
    create_parametrization_from_named_scenarios,
)
from src.simulation.load_simulation_inputs import get_simulation_dependencies
from src.simulation.load_simulation_inputs import load_simulation_inputs


SPRING_START = "2021-02-15"

DEPENDENCIES = get_simulation_dependencies(debug=FAST_FLAG == "debug")


if FAST_FLAG == "debug":
    n_baseline_seeds = 1
    n_main_scenario_seeds = 0
    n_side_scenario_seeds = 0
elif FAST_FLAG == "verify":  # use 27 cores -> 2 rounds
    n_baseline_seeds = 10  # 2x
    n_main_scenario_seeds = 3  # 4x
    n_side_scenario_seeds = 1  # 11
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
    "start_date": SPRING_START,
    "end_date": "2021-05-16" if FAST_FLAG != "debug" else "2021-05-01",
}

summer_dates = {
    "start_date": SPRING_START,
    "end_date": "2021-07-01" if FAST_FLAG != "debug" else "2021-06-01",
}

NAMED_SCENARIOS = {
    # Baseline Scenarios
    "fall_baseline": {
        "sim_input_scenario": "baseline",
        "params_scenario": "baseline",
        "start_date": "2020-10-15",
        "end_date": "2020-12-23" if FAST_FLAG != "debug" else "2020-11-15",
        "n_seeds": n_baseline_seeds,
    },
    "summer_baseline": {
        "sim_input_scenario": "baseline",
        "params_scenario": "baseline",
        "n_seeds": n_baseline_seeds,
        **summer_dates,
    },
    # Policy Scenarios
    "spring_without_vaccines": {
        "sim_input_scenario": "no_vaccinations_after_feb_15",
        "params_scenario": "baseline",
        "n_seeds": n_main_scenario_seeds,
        **spring_dates,
    },
    "spring_with_more_vaccines": {
        "sim_input_scenario": "vaccinations_after_easter_as_on_strongest_week_day",
        "params_scenario": "baseline",
        "n_seeds": n_side_scenario_seeds,
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
    # Note this scenario does still assume that only 70% of employers comply, i.e.
    # it ensures that 70% of workers get regularly tested. Given that the share of
    # workers accepting a rapid test from their employer is time invariant this also
    # means that before Easter there is already a lot more testing going on.
    "spring_with_obligatory_work_rapid_tests": {
        "sim_input_scenario": "baseline",
        "params_scenario": "obligatory_rapid_tests_for_employees",
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
    # For the school opening scenarios we assume that the supply of rapid tests is large
    # enough to allow the same fraction of individuals that should be tested is actually
    # tested. After Easter that was 95% for educ workers and 75% for school pupils and
    # increases from there to 1 for pupils.
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
    # Summer Scenarios
    "summer_educ_open": {
        "sim_input_scenario": "open_all_educ_after_summer_scenario_start",
        "params_scenario": "baseline",
        "n_seeds": n_main_scenario_seeds,
        **summer_dates,
    },
    "summer_reduced_test_demand": {
        "sim_input_scenario": "baseline",
        "params_scenario": "reduce_rapid_test_demand_after_summer_scenario_start_by_half",  # noqa: E501
        "n_seeds": n_main_scenario_seeds,
        **summer_dates,
    },
    "summer_strict_home_office": {
        "sim_input_scenario": "strict_home_office_after_summer_scenario_start",
        "params_scenario": "baseline",
        "n_seeds": 0,
        **summer_dates,
    },
    "summer_more_rapid_tests_at_work": {
        "sim_input_scenario": "baseline",
        "params_scenario": "rapid_test_with_90pct_compliance_after_summer_scenario_start",  # noqa: E501
        "n_seeds": n_side_scenario_seeds,
        **summer_dates,
    },
    "summer_optimistic_vaccinations": {
        "sim_input_scenario": "vaccinations_after_summer_scenario_start_as_on_strongest_week_day",  # noqa: E501
        "params_scenario": "baseline",
        "n_seeds": n_side_scenario_seeds,
        **summer_dates,
    },
}

SCENARIOS = create_parametrization_from_named_scenarios(NAMED_SCENARIOS, FAST_FLAG)


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
