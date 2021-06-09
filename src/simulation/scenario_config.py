import pandas as pd

from src.config import BLD
from src.config import FAST_FLAG

SPRING_START = pd.Timestamp("2021-01-01")


NON_INCIDENCE_OUTCOMES = [
    "ever_vaccinated",
    "r_effective",
    "share_ever_rapid_test",
    "share_rapid_test_in_last_week",
    "share_b117",
    "share_doing_rapid_test_today",
]


def create_path_to_last_states_of_simulation(name, seed):
    path = BLD / "simulations" / "last_states" / f"{FAST_FLAG}_{name}_{seed}.pkl"
    return path


def create_path_to_period_outputs_of_simulation(name, seed):
    """Return the path to the simulation results with the period outcomes."""
    path = BLD / "simulations" / "period_outputs" / f"{FAST_FLAG}_{name}_{seed}.pkl"
    return path


def create_path_to_share_known_cases_of_scenario(name, groupby):
    if groupby is None:
        file_name = f"{FAST_FLAG}_{name}_by_{groupby}.pkl"
    else:
        file_name = f"{FAST_FLAG}_{name}.pkl"
    return BLD / "simulations" / "share_known_cases" / file_name


def create_path_to_scenario_outcome_time_series(scenario_name, entry):
    weekly = not any(outcome in entry for outcome in NON_INCIDENCE_OUTCOMES)
    if weekly:
        file_name = f"{FAST_FLAG}_{entry}_weekly_incidence.pkl"
    elif "ever_vaccinated" in entry:
        file_name = f"{FAST_FLAG}_{entry}_share.pkl"
    else:
        file_name = f"{FAST_FLAG}_{entry}.pkl"
    return BLD / "simulations" / "time_series" / scenario_name / file_name


def create_path_to_share_known_cases_plot(name, groupby):
    if groupby is not None:
        fig_name = f"{FAST_FLAG}_{name}_by_{groupby}.pdf"
    else:
        fig_name = f"{FAST_FLAG}_{name}.pdf"
    return BLD / "figures" / "share_known_cases" / fig_name


def create_path_to_group_incidence_plot(name, outcome, groupby):
    fig_name = f"{FAST_FLAG}_{name}_{outcome}.pdf"
    return BLD / "figures" / "incidences_by_group" / groupby / fig_name


def create_path_for_weekly_outcome_of_scenario(
    comparison_name, fast_flag, outcome, suffix
):
    file_name = f"{fast_flag}_{outcome}.{suffix}"
    if suffix == "pdf":
        path = BLD / "figures" / "scenario_comparisons" / comparison_name / file_name
    elif suffix == "csv":
        path = BLD / "tables" / "scenario_comparisons" / comparison_name / file_name
    else:
        raise ValueError(f"Unknown suffix {suffix}. Only 'pdf' and 'csv' supported")
    return path


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
        n_main_seeds = 1
        n_other_seeds = 1
    elif FAST_FLAG == "verify":
        n_main_seeds = 1
        n_other_seeds = 1
    elif FAST_FLAG == "full":
        n_main_seeds = 20
        n_other_seeds = 20
    else:
        raise ValueError(
            f"Unknown FAST_FLAG {FAST_FLAG}."
            "Only 'debug', 'verify' or 'full' are allowed."
        )

    if FAST_FLAG != "debug":
        fall_dates = {
            "start_date": "2020-09-15",
            "end_date": SPRING_START - pd.Timedelta(days=1),
        }
        spring_dates = {"start_date": SPRING_START, "end_date": "2021-05-31"}
        combined_dates = {
            "start_date": fall_dates["start_date"],
            "end_date": spring_dates["end_date"],
        }
        summer_dates = {"start_date": "2021-06-01", "end_date": "2021-07-25"}
    else:
        # for the plotting we need that combined and spring have dates after 2021-01-15
        combined_dates = {"start_date": "2020-12-30", "end_date": "2021-01-18"}
        fall_dates = {
            "start_date": "2020-12-25",
            "end_date": "2020-12-31",
        }
        spring_dates = {
            "start_date": "2021-01-01",
            "end_date": "2021-01-18",
        }
        summer_dates = {
            "start_date": "2021-01-19",
            "end_date": "2021-01-24",
        }

    named_scenarios = {
        # Baseline Scenarios
        "combined_baseline": {
            "sim_input_scenario": "baseline",
            "params_scenario": "baseline",
            "n_seeds": n_main_seeds,
            "save_last_states": True,
            "is_resumed": False,
            **combined_dates,
        },
        "fall_baseline": {
            "sim_input_scenario": "baseline",
            "params_scenario": "baseline",
            "n_seeds": n_main_seeds,
            "save_last_states": True,
            "is_resumed": False,
            **fall_dates,
        },
        "spring_baseline": {
            "sim_input_scenario": "baseline",
            "params_scenario": "baseline",
            "n_seeds": n_main_seeds,
            "save_last_states": True,
            **spring_dates,
        },
        "summer_baseline": {
            "sim_input_scenario": "baseline",
            "params_scenario": "baseline",
            "n_seeds": n_main_seeds,
            "save_last_states": True,
            "is_resumed": "combined",
            **summer_dates,
        },
        # Scenarios for the main plots
        "spring_no_effects": {
            "sim_input_scenario": "just_seasonality",
            "params_scenario": "no_seasonality",
            "n_seeds": n_main_seeds,
            **spring_dates,
        },
        "spring_without_seasonality": {
            "sim_input_scenario": "baseline",
            "params_scenario": "no_seasonality",
            "n_seeds": n_main_seeds,
            **spring_dates,
        },
        "spring_without_vaccines": {
            "sim_input_scenario": "no_vaccinations",
            "params_scenario": "baseline",
            "n_seeds": n_main_seeds,
            **spring_dates,
        },
        "spring_without_rapid_tests": {
            "sim_input_scenario": "no_rapid_tests",
            "params_scenario": "baseline",
            "n_seeds": n_main_seeds,
            **spring_dates,
        },
        # School Scenarios
        "spring_close_educ_after_easter": {
            "sim_input_scenario": "close_educ_after_april_5",
            "params_scenario": "no_rapid_tests_at_schools_after_easter",
            "n_seeds": n_main_seeds,
            **spring_dates,
        },
        # For the school opening scenarios we assume that the supply of rapid tests is
        # large enough to allow the same fraction of individuals that should be tested
        # is actually tested. After Easter that was 95% for educ workers and 75% for
        # school pupils and increases from there to 1 for pupils.
        "spring_educ_open_after_easter_with_tests": {
            "sim_input_scenario": "open_all_educ_after_easter",
            "params_scenario": "baseline",
            "n_seeds": n_main_seeds,
            **spring_dates,
        },
        "spring_educ_open_after_easter_without_tests": {
            "sim_input_scenario": "open_all_educ_after_easter",
            "params_scenario": "no_rapid_tests_at_schools_after_easter",
            "n_seeds": n_main_seeds,
            **spring_dates,
        },
        # Other Scenarios
        "spring_vaccinate_1_pct_per_day_after_easter": {
            "sim_input_scenario": "vaccinate_1_pct_per_day_after_easter",
            "params_scenario": "baseline",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "spring_without_school_rapid_tests": {
            "sim_input_scenario": "baseline",
            "params_scenario": "no_rapid_tests_at_schools",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "spring_without_work_rapid_tests": {
            "sim_input_scenario": "baseline",
            "params_scenario": "no_rapid_tests_at_work",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "spring_without_private_rapid_tests": {
            "sim_input_scenario": "baseline",
            "params_scenario": "no_private_rapid_test_demand",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "spring_without_school_and_work_rapid_tests": {
            "sim_input_scenario": "baseline",
            "params_scenario": "no_rapid_tests_at_schools_and_work",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "spring_without_school_and_private_rapid_tests": {
            "sim_input_scenario": "baseline",
            "params_scenario": "no_rapid_tests_at_schools_and_private",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "spring_without_work_and_private_rapid_tests": {
            "sim_input_scenario": "baseline",
            "params_scenario": "no_rapid_tests_at_work_and_private",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "spring_without_rapid_tests_and_no_vaccinations": {
            "sim_input_scenario": "just_seasonality",
            "params_scenario": "baseline",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "spring_without_rapid_tests_without_seasonality": {  # i.e. only vaccinations
            "sim_input_scenario": "no_rapid_tests",
            "params_scenario": "no_seasonality",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "spring_without_vaccinations_without_seasonality": {  # i.e. only rapid tests
            "sim_input_scenario": "no_vaccinations",
            "params_scenario": "no_seasonality",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "spring_start_all_rapid_tests_after_easter": {
            "sim_input_scenario": "baseline",
            "params_scenario": "start_all_rapid_tests_after_easter",
            "n_seeds": n_other_seeds,
            **spring_dates,
        },
        "summer_strict_home_office_continue_testing": {
            "sim_input_scenario": "strict_home_office_after_summer_scenario_start",
            "params_scenario": "baseline",
            "n_seeds": n_other_seeds,
            "is_resumed": "combined",
            **summer_dates,
        },
        "summer_strict_home_office_reduce_testing": {
            "sim_input_scenario": "strict_home_office_after_summer_scenario_start",
            "params_scenario": "reduce_work_rapid_test_demand_after_summer_scenario_start_by_half",  # noqa: E501
            "n_seeds": n_other_seeds,
            "is_resumed": "combined",
            **summer_dates,
        },
        "summer_normal_home_office_reduce_testing": {
            "sim_input_scenario": "baseline",
            "params_scenario": "reduce_work_rapid_test_demand_after_summer_scenario_start_by_half",  # noqa: E501
            "n_seeds": n_other_seeds,
            "is_resumed": "combined",
            **summer_dates,
        },
    }
    return named_scenarios


def get_available_scenarios(named_scenarios):
    available_scenarios = sorted(
        name for name, spec in named_scenarios.items() if spec["n_seeds"] > 0
    )
    return available_scenarios
