import pandas as pd

from src.config import AFTER_EASTER
from src.plotting.plotting import BLUE
from src.plotting.plotting import BROWN
from src.plotting.plotting import GREEN
from src.plotting.plotting import ORANGE
from src.plotting.plotting import PURPLE
from src.plotting.plotting import RED
from src.plotting.plotting import TEAL

WITH_VACCINATIONS = r"vaccinations $\checkmark$ "
NO_VACCINATIONS = r"vaccinations $\times$ "

WITH_SEASONALITY = r"seasonality $\checkmark$ "
NO_SEASONALITY = r"seasonality $\times$ "

WITH_RAPID_TESTS = r"rapid tests $\checkmark$ "
NO_RAPID_TESTS = r"rapid tests $\times$ "


SCHOOL_SCENARIOS = [
    "spring_educ_open_after_easter_without_tests",
    "spring_educ_open_after_easter_with_tests",
    "spring_close_educ_after_easter",
    "spring_baseline",
]


DECOMPOSITION_PLOTS = {
    "effect_of_channels_on_pessimistic_scenario": {
        "title": "Effect on {outcome} when Adding "
        "Single Channels\non the Pessimistic Scenario",
        "scenarios": [
            "spring_baseline",
            "spring_no_effects",
            "spring_without_rapid_tests_without_seasonality",  # just vaccinations
            "spring_without_rapid_tests_and_no_vaccinations",  # just seasonality
            "spring_without_vaccinations_without_seasonality",  # just rapid tests
        ],
        "name_to_label": {
            "spring_no_effects": NO_VACCINATIONS + NO_SEASONALITY + NO_RAPID_TESTS,
            "spring_without_rapid_tests_and_no_vaccinations": NO_VACCINATIONS
            + WITH_SEASONALITY
            + NO_RAPID_TESTS,  # just seasonality
            "spring_without_rapid_tests_without_seasonality": WITH_VACCINATIONS
            + NO_SEASONALITY
            + NO_RAPID_TESTS,  # just vaccinations
            "spring_without_vaccinations_without_seasonality": NO_VACCINATIONS
            + NO_SEASONALITY
            + WITH_RAPID_TESTS,  # just rapid tests
            "spring_baseline": WITH_VACCINATIONS + WITH_SEASONALITY + WITH_RAPID_TESTS,
        },
        "colors": [BLUE, RED, ORANGE, GREEN, PURPLE],
        "plot_start": pd.Timestamp("2021-01-15"),
    },
    "one_off_and_combined": {
        "title": "The Effect of Each Channel on {outcome} Separately",
        "scenarios": [
            "spring_baseline",
            "spring_no_effects",
            "spring_without_seasonality",
            "spring_without_vaccines",
            "spring_without_rapid_tests",
        ],
        "name_to_label": {
            "spring_no_effects": NO_VACCINATIONS + NO_SEASONALITY + NO_RAPID_TESTS,
            "spring_without_seasonality": WITH_VACCINATIONS
            + NO_SEASONALITY
            + WITH_RAPID_TESTS,
            "spring_without_vaccines": NO_VACCINATIONS
            + WITH_SEASONALITY
            + WITH_RAPID_TESTS,
            "spring_without_rapid_tests": WITH_VACCINATIONS
            + WITH_SEASONALITY
            + NO_RAPID_TESTS,
            "spring_baseline": WITH_VACCINATIONS + WITH_SEASONALITY + WITH_RAPID_TESTS,
        },
        "colors": [BLUE, RED, GREEN, TEAL, ORANGE],
        "plot_start": pd.Timestamp("2021-01-15"),
    },
    "school_scenarios": {
        "title": "The Effect of Schools on {outcome}",
        "scenarios": SCHOOL_SCENARIOS,
        "name_to_label": {
            "spring_educ_open_after_easter_without_tests": "open schools without tests",
            "spring_educ_open_after_easter_with_tests": "open schools with tests",
            "spring_close_educ_after_easter": "keep schools closed",
            "spring_baseline": "enacted school policies",
        },
        "colors": [PURPLE, RED, BROWN, BLUE],
        "plot_start": AFTER_EASTER,
    },
    # Other Fixed Plots
    "effect_of_rapid_tests": {
        "title": "Decomposing the Effect of Rapid Tests on {outcome}",
        "scenarios": [
            "spring_without_rapid_tests",
            "spring_without_work_rapid_tests",
            "spring_without_school_rapid_tests",
            "spring_without_private_rapid_tests",
            "spring_baseline",
        ],
        "colors": [BROWN, RED, ORANGE, PURPLE, BLUE],
        "name_to_label": {
            "spring_without_rapid_tests": "no rapid tests",
            "spring_without_school_rapid_tests": "without school rapid tests",
            "spring_without_work_rapid_tests": "without work rapid tests",
            "spring_baseline": "full rapid test demand",
            "spring_without_private_rapid_tests": "without private rapid test demand",
        },
        "plot_start": pd.Timestamp("2021-01-15"),
    },
    "explaining_the_decline": {
        "title": "Explaining the Puzzling Decline in\n{outcome}",
        "scenarios": [
            "spring_no_effects",
            "spring_without_rapid_tests_and_no_vaccinations",
            "spring_without_rapid_tests",
            "spring_baseline",
        ],
        "colors": [RED, GREEN, ORANGE, BLUE],
        "name_to_label": {
            "spring_no_effects": "pessimistic scenario",
            "spring_without_rapid_tests_and_no_vaccinations": NO_VACCINATIONS
            + WITH_SEASONALITY
            + NO_RAPID_TESTS,
            "spring_without_rapid_tests": WITH_VACCINATIONS
            + WITH_SEASONALITY
            + NO_RAPID_TESTS,
            "spring_baseline": WITH_VACCINATIONS + WITH_SEASONALITY + WITH_RAPID_TESTS,
        },
        "plot_start": pd.Timestamp("2021-01-15"),
    },
    # Variable Plots
    "pessimistic_scenario": {
        "title": "Replicating the Pessimistic Scenarios of March",
        "scenarios": ["spring_no_effects"],
        "colors": [RED],
        "plot_start": pd.Timestamp("2021-01-15"),
    },
    "vaccine_scenarios": {
        "title": "Effect of Different Vaccination Scenarios on {outcome}",
        "scenarios": [
            "spring_baseline",
            "spring_vaccinate_1_pct_per_day_after_easter",
            "spring_without_vaccines",
        ],
        "colors": [BLUE, GREEN, RED],
        "name_to_label": {
            "spring_baseline": "current vaccination progress",
            "spring_vaccinate_1_pct_per_day_after_easter": "vaccinate 1 percent "
            "of the\npopulation every day after Easter",
            "spring_without_vaccines": "stop vaccinations on February 10th",
        },
        "plot_start": AFTER_EASTER - pd.Timedelta(days=14),
        "scenario_starts": ([(AFTER_EASTER, "start of increased vaccinations")]),
    },
    "illustrate_rapid_tests": {
        "title": "Illustrate the Effect of Rapid Tests on {outcome}",
        "scenarios": [
            "spring_baseline",
            "spring_without_rapid_tests",
            "spring_start_all_rapid_tests_after_easter",
        ],
        "colors": [BLUE, PURPLE, ORANGE],
        "name_to_label": {
            "spring_baseline": "calibrated rapid test scenario",
            "spring_without_rapid_tests": "no rapid tests",
            "spring_start_all_rapid_tests_after_easter": "start rapid tests at Easter",
        },
        "plot_start": AFTER_EASTER - pd.Timedelta(days=14),
        "scenario_starts": ([(AFTER_EASTER, "Easter")]),
    },
    "new_work_scenarios": {
        "title": "The Effect of Home Office and Work Rapid Tests on {outcome}",
        "scenarios": [
            "spring_baseline",
            "spring_reduce_work_test_offers_to_23_pct_after_easter",
            "spring_mandatory_work_rapid_tests_after_easter",
            "spring_10_pct_less_work_in_person_after_easter",
            "spring_10_pct_more_work_in_person_after_easter",
        ],
        "colors": None,
        "name_to_label": {
            "spring_baseline": "baseline",
            "spring_reduce_work_test_offers_to_23_pct_after_easter": "rapid tests as "
            "in mid March",
            "spring_mandatory_work_rapid_tests_after_easter": "mandatory rapid tests",
            "spring_10_pct_less_work_in_person_after_easter": "10% less presence "
            "at workplace",
            "spring_10_pct_more_work_in_person_after_easter": "10% more presence\n"
            "at workplace",
        },
        "plot_start": AFTER_EASTER,
    },
    "add_single_rapid_test_chanel_to_pessimistic": {
        "title": "",
        "scenarios": [
            "spring_without_rapid_tests",
            "spring_without_school_and_work_rapid_tests",  # just private
            "spring_without_school_and_private_rapid_tests",  # just work
            "spring_without_work_and_private_rapid_tests",  # just school
        ],
        "colors": [PURPLE, GREEN, RED, ORANGE],
        "name_to_label": {
            "spring_without_rapid_tests": "no rapid tests at all",
            "spring_without_school_and_work_rapid_tests": "just private rapid tests",
            "spring_without_school_and_private_rapid_tests": "just work rapid tests",
            "spring_without_work_and_private_rapid_tests": "just school rapid tests",
        },
    },
    "random_rapid_tests_vs_baseline": {
        "title": "",
        "scenarios": [
            "spring_baseline",
            "spring_with_completely_random_rapid_tests",
            "spring_with_random_rapid_tests_with_30pct_refusers",
        ],
        "colors": [BLUE, RED, PURPLE],
        "plot_start": AFTER_EASTER,
    },
    "robustness_check": {
        "title": "",
        "scenarios": [
            "spring_baseline",
            "robustness_check_early",
            "robustness_check_medium",
            "robustness_check_late",
        ],
        "colors": [BLUE, ORANGE, RED, BROWN],
        "name_to_label": {
            "spring_baseline": "ex post",
            "robustness_check_early": "full rapid test availability: May 1",
            "robustness_check_medium": "full rapid test availability: May 20",
            "robustness_check_late": "full rapid test availability: June 10",
        },
    },
}
