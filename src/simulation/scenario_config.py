from src.config import BLD
from src.config import FAST_FLAG

SPRING_START = "2021-02-15"


def create_path_to_last_states_of_simulation(name, seed):
    path = (
        BLD
        / "simulations"
        / f"{FAST_FLAG}_{name}_{seed}"
        / "last_states"
        / "last_states.parquet"
    )
    return path


def create_path_to_initial_group_share_known_cases(name, date):
    file_name = f"{FAST_FLAG}_{name}_for_{date.date()}.pkl"
    return BLD / "simulations" / "share_known_case_prediction" / file_name


def create_path_to_share_known_cases_of_scenario(name):
    file_name = f"{FAST_FLAG}_{name}.pkl"
    return BLD / "simulations" / "share_known_cases" / file_name


def create_path_to_weekly_outcome_of_scenario(name, outcome, groupby):
    if groupby is None:
        file_name = f"{FAST_FLAG}_{name}_{outcome}.pkl"
    else:
        file_name = f"{FAST_FLAG}_{name}_{outcome}_by_{groupby}.pkl"
    return BLD / "simulations" / "incidences" / file_name


def create_path_to_share_known_cases_plot(name, groupby):
    if groupby is not None:
        fig_name = f"{FAST_FLAG}_{name}_by_{groupby}.png"
    else:
        fig_name = f"{FAST_FLAG}_{name}.png"
    return BLD / "figures" / "share_known_cases" / fig_name


# =============================================================================


def get_named_scenarios():
    """Get the named scenarios.

    Returns:
        dict: Nested dictionary. The outer keys are the names of the scenarios. The
            inner dictionary are the specs passed to load_simulation_inputs and contains
            'start_date', 'end_date', 'sim_input_scenario', 'params_scenario' and
            'n_seeds'.

    """
    if FAST_FLAG == "debug":
        n_baseline_seeds = 1
        n_main_scenario_seeds = 1
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

    named_scenarios = {
        # Baseline Scenarios
        "fall_baseline": {
            "sim_input_scenario": "baseline",
            "params_scenario": "baseline",
            "start_date": "2020-10-15",
            "end_date": "2020-12-23",
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
        "spring_without_school_rapid_tests": {
            "sim_input_scenario": "baseline",
            "params_scenario": "no_rapid_tests_at_schools",
            "n_seeds": n_side_scenario_seeds,
            **spring_dates,
        },
        "spring_without_work_rapid_tests": {
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
        "spring_with_mandatory_work_rapid_tests": {
            "sim_input_scenario": "baseline",
            "params_scenario": "obligatory_rapid_tests_for_employees",
            "n_seeds": n_side_scenario_seeds,
            **spring_dates,
        },
        # Rapid Tests vs School Closures
        "spring_emergency_care_after_easter_without_school_rapid_tests": {
            "sim_input_scenario": "only_strict_emergency_care_after_april_5",
            "params_scenario": "no_rapid_tests_at_schools",
            "n_seeds": n_side_scenario_seeds,
            **spring_dates,
        },
        # For the school opening scenarios we assume that the supply of rapid tests is
        # large enough to allow the same fraction of individuals that should be tested
        # is actually tested. After Easter that was 95% for educ workers and 75% for
        # school pupils and increases from there to 1 for pupils.
        "spring_educ_open_after_easter": {
            "sim_input_scenario": "open_all_educ_after_easter",
            "params_scenario": "baseline",
            "n_seeds": n_main_scenario_seeds,
            **spring_dates,
        },
        "spring_open_educ_after_easter_with_tests_every_other_day": {
            "sim_input_scenario": "open_all_educ_after_easter",
            "params_scenario": "rapid_tests_at_school_every_other_day_after_april_5",
            "n_seeds": n_side_scenario_seeds,
            **spring_dates,
        },
        "spring_open_educ_after_easter_with_daily_tests": {
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
    return named_scenarios


def get_available_scenarios(named_scenarios):
    available_scenarios = {
        name for name, spec in named_scenarios.items() if spec["n_seeds"] > 0
    }
    return available_scenarios
