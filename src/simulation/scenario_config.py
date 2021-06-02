import pandas as pd

from src.config import BLD
from src.config import FAST_FLAG

SPRING_START = pd.Timestamp("2021-01-01")


NON_INCIDENCE_OUTCOMES = [
    "ever_vaccinated",
    "r_effective",
    "share_ever_rapid_test",
    "share_rapid_test_in_last_week",
]


def create_path_to_last_states_of_simulation(name, seed):
    path = BLD / "simulations" / "last_states" / f"{FAST_FLAG}_{name}_{seed}.pkl"
    return path


def create_path_to_period_outputs_of_simulation(name, seed):
    """Return the path to the simulation results with the period outcomes."""
    path = BLD / "simulations" / "period_outputs" / f"{FAST_FLAG}_{name}_{seed}.pkl"
    return path


def create_path_to_share_known_cases_of_scenario(name):
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
        fig_name = f"{FAST_FLAG}_{name}_by_{groupby}.png"
    else:
        fig_name = f"{FAST_FLAG}_{name}.png"
    return BLD / "figures" / "share_known_cases" / fig_name


def create_path_to_group_incidence_plot(name, outcome, groupby):
    fig_name = f"{FAST_FLAG}_{name}_{outcome}.png"
    return BLD / "figures" / "incidences_by_group" / groupby / fig_name


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
        n_main_seeds = 2
        n_other_seeds = 2
    elif FAST_FLAG == "verify":
        n_main_seeds = 15
        n_other_seeds = 0
    elif FAST_FLAG == "full":
        n_main_seeds = 25
        n_other_seeds = 25
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
    else:
        fall_dates = {
            "start_date": "2020-12-25",
            "end_date": "2020-12-31",
        }
        spring_dates = {
            "start_date": "2021-01-01",
            "end_date": "2021-01-05",
        }
        combined_dates = {"start_date": "2020-12-25", "end_date": "2021-01-05"}

    named_scenarios = {
        # Baseline Scenarios
        "combined_baseline": {
            "sim_input_scenario": "baseline",
            "params_scenario": "baseline",
            "n_seeds": n_main_seeds,
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
            **spring_dates,
        },
        # Scenarios for the main plots
        "spring_no_effects": {
            "sim_input_scenario": "no_rapid_tests_and_no_vaccinations",
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
            "params_scenario": "no_rapid_tests_at_schools",
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
            "params_scenario": "no_rapid_tests_at_schools",
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
        "spring_without_rapid_tests_and_no_vaccinations": {
            "sim_input_scenario": "no_rapid_tests_and_no_vaccinations",
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
    }
    return named_scenarios


def get_available_scenarios(named_scenarios):
    available_scenarios = sorted(
        name for name, spec in named_scenarios.items() if spec["n_seeds"] > 0
    )
    return available_scenarios
