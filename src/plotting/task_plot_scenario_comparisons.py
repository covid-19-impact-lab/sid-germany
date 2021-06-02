import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import FAST_FLAG
from src.config import SRC
from src.plotting.plotting import create_automatic_labels
from src.plotting.plotting import plot_incidences
from src.plotting.plotting import shorten_dfs
from src.policies.policy_tools import filter_dictionary
from src.simulation.scenario_config import create_path_to_scenario_outcome_time_series
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios

_MODULE_DEPENDENCIES = {
    "plotting.py": SRC / "plotting" / "plotting.py",
    "policy_tools.py": SRC / "policies" / "policy_tools.py",
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
    "cosmo_frequency": SRC
    / "original_data"
    / "testing"
    / "cosmo_selftest_frequency_last_four_weeks.csv",
    "cosmo_ever_rapid_test": SRC
    / "original_data"
    / "testing"
    / "cosmo_share_ever_had_a_rapid_test.csv",
}

NAMED_SCENARIOS = get_named_scenarios()

AFTER_EASTER = pd.Timestamp("2021-04-06")

SCHOOL_SCENARIOS = [
    "spring_educ_open_after_easter_without_tests",
    "spring_educ_open_after_easter_with_tests",
    "spring_close_educ_after_easter",
]

# Colors
BLUE = "#4e79a7"
ORANGE = "#f28e2b"
RED = "#e15759"
TEAL = "#76b7b2"
GREEN = "#59a14f"
YELLOW = "#edc948"
PURPLE = "#b07aa1"
BROWN = "#9c755f"


OUTCOMES = [
    "newly_infected",
    "new_known_case",
    "newly_deceased",
    "r_effective",
    "share_ever_rapid_test",
    "share_rapid_test_in_last_week",
]

PLOTS = {
    # Main Plots (Fixed)
    "fitness_plot": {
        "title": "Simulated versus Empirical Infections",
        "scenarios": ["combined_baseline"],
        "name_to_label": {"combined_baseline": "simulated"},
        "colors": [BLUE],
    },
    "fall_fit": {
        "title": "{outcome} in Fall",
        "scenarios": ["fall_baseline"],
        "name_to_label": {"fall_baseline": "simulation"},
        "colors": [BLUE],
    },
    "spring_fit": {
        "title": "{outcome} in Spring",
        "scenarios": ["spring_baseline"],
        "name_to_label": {"spring_baseline": "simulation"},
        "colors": [BLUE],
    },
    "one_off_and_combined": {
        "title": "The Effect of Each Channel on {outcome} Separately",
        "scenarios": [
            "spring_no_effects",
            "spring_without_seasonality",
            "spring_without_vaccines",
            "spring_without_rapid_tests",
            "spring_baseline",
        ],
        "name_to_label": {
            "spring_no_effects": "without any channel",
            "spring_without_seasonality": "without seasonality",
            "spring_without_vaccines": "without vaccines",
            "spring_without_rapid_tests": "without rapid tests",
            "spring_baseline": "with all channels",
        },
        "colors": None,
    },
    "school_scenarios": {
        "title": "The Effect of Schools on {outcome}",
        "scenarios": SCHOOL_SCENARIOS,
        "name_to_label": {
            "spring_educ_open_after_easter_without_tests": "open schools without tests",
            "spring_educ_open_after_easter_with_tests": "open schools with tests",
            "spring_close_educ_after_easter": "keep schools closed",
        },
        "colors": None,
        "plot_start": AFTER_EASTER,
    },
    # Other Fixed Plots
    "effect_of_rapid_tests": {
        "title": "Decomposing the Effect of Rapid Tests on {outcome}",
        "scenarios": [
            "spring_without_rapid_tests",
            "spring_without_work_rapid_tests",
            "spring_without_school_rapid_tests",
            "spring_baseline",
        ],
        "name_to_label": {
            "spring_without_rapid_tests": "no rapid tests",
            "spring_without_school_rapid_tests": "without school rapid tests",
            "spring_without_work_rapid_tests": "without work rapid tests",
            "spring_baseline": "work, school and private rapid tests",
        },
        "colors": [BROWN, RED, ORANGE, BLUE],
    },
    "explaining_the_decline": {
        "title": "Explaining the Puzzling Decline in\n{outcome}",
        "scenarios": [
            "spring_no_effects",
            "spring_without_rapid_tests_and_no_vaccinations",
            "spring_without_rapid_tests",
            "spring_baseline",
        ],
        "name_to_label": {
            "spring_no_effects": "pessimistic scenario",
            "spring_without_rapid_tests_and_no_vaccinations": "with seasonality only",
            "spring_without_rapid_tests": "with seasonality and vaccinations",
            "spring_baseline": "with seasonality, vaccinations\nand rapid tests",
        },
        "colors": None,
    },
    # Variable Plots
    "pessimistic_scenario": {
        "title": "Replicating the Pessimistic Scenarios of March",
        "scenarios": ["spring_no_effects"],
        "colors": None,
    },
    "vaccine_scenarios": {
        "title": "Effect of Different Vaccination Scenarios on {outcome}",
        "scenarios": [
            "spring_baseline",
            "spring_vaccinate_1_pct_per_day_after_easter",
            "spring_without_vaccines",
        ],
        "name_to_label": {
            "spring_baseline": "current vaccination progress",
            "spring_vaccinate_1_pct_per_day_after_easter": "vaccinate 1 percent "
            "of the\npopulation every day after Easter",
            "spring_without_vaccines": "stop vaccinations on February 10th",
        },
        "colors": [BLUE, GREEN, RED],
        "scenario_starts": ([(AFTER_EASTER, "start of increased vaccinations")]),
    },
    "illustrate_rapid_tests": {
        "title": "Illustrate the Effect of Rapid Tests on {outcome}",
        "scenarios": [
            "spring_baseline",
            "spring_without_rapid_tests",
            "spring_start_all_rapid_tests_after_easter",
        ],
        "name_to_label": {
            "spring_baseline": "calibrated rapid test scenario",
            "spring_without_rapid_tests": "no rapid tests",
            "spring_start_all_rapid_tests_after_easter": "start rapid tests at Easter",
        },
        "colors": [BLUE, PURPLE, RED],
        "scenario_starts": ([(AFTER_EASTER, "Easter")]),
    },
    "effect_of_channels_on_pessimistic_scenario": {
        "title": "Effect on {outcome} when Adding "
        "Single Channels\non the Pessimistic Scenario",
        "scenarios": [
            "spring_no_effects",
            "spring_without_rapid_tests_and_no_vaccinations",
            "spring_without_rapid_tests_without_seasonality",
            "spring_without_vaccinations_without_seasonality",
        ],
        "name_to_label": {
            "spring_no_effects": "pessimistic scenario",
            "spring_without_rapid_tests_and_no_vaccinations": "just seasonality",
            "spring_without_rapid_tests_without_seasonality": "just vaccinations",
            "spring_without_vaccinations_without_seasonality": "just rapid tests",
        },
        "colors": None,
        "empirical": False,
    },
}
"""Dict[str, Dict[str, str]]: A dictionary containing the plots to create.

Each key in the dictionary is a name for a collection of scenarios. The values are
dictionaries with the title and the lists of scenario names which are combined to
create the collection.

"""

AVAILABLE_SCENARIOS = get_available_scenarios(NAMED_SCENARIOS)

plotted_scenarios = {x for spec in PLOTS.values() for x in spec["scenarios"]}
assert set(AVAILABLE_SCENARIOS).issubset(
    plotted_scenarios
), "The following scenarios do not appear in any plots: " + "\n\t".join(
    list(set(AVAILABLE_SCENARIOS).difference(plotted_scenarios))
)


def create_path_for_weekly_outcome_of_scenario(
    comparison_name, fast_flag, outcome, suffix
):
    file_name = f"{fast_flag}_{outcome}.{suffix}"
    if suffix == "png":
        path = BLD / "figures" / "comparisons" / comparison_name / file_name
    elif suffix == "csv":
        path = BLD / "tables" / "comparisons" / comparison_name / file_name
    else:
        raise ValueError(f"Unknown suffix {suffix}. Only 'png' and 'csv' supported")
    return path


def create_parametrization(plots, named_scenarios, fast_flag, outcomes):
    available_scenarios = get_available_scenarios(named_scenarios)
    parametrization = []
    for outcome in outcomes:
        for comparison_name, plot_info in plots.items():
            title = plot_info["title"]

            # need to keep the right colors
            scenarios = []
            colors = [] if plot_info["colors"] is not None else None
            for i, name in enumerate(plot_info["scenarios"]):
                if name in available_scenarios:
                    scenarios.append(name)
                    if colors is not None:
                        colors.append(plot_info["colors"][i])

            depends_on = {
                scenario_name: create_path_to_scenario_outcome_time_series(
                    scenario_name=scenario_name, entry=outcome
                )
                for scenario_name in scenarios
            }

            missing_scenarios = set(depends_on) - set(named_scenarios)
            if missing_scenarios:
                raise ValueError(f"Some scenarios are missing: {missing_scenarios}.")

            produces = {
                "fig": create_path_for_weekly_outcome_of_scenario(
                    comparison_name, fast_flag, outcome, "png"
                ),
                "data": create_path_for_weekly_outcome_of_scenario(
                    comparison_name, fast_flag, outcome, "csv"
                ),
            }
            # only create a plot if at least one scenario had a seed.
            if depends_on:
                parametrization.append(
                    (
                        depends_on,
                        outcome,
                        title,
                        colors,
                        plot_info.get("name_to_label"),
                        plot_info.get("scenario_starts"),
                        plot_info.get("plot_start"),
                        plot_info.get("empirical"),
                        produces,
                    )
                )

    return (
        "depends_on, outcome, title, colors, name_to_label, scenario_starts, "
        "plot_start, empirical, produces",
        parametrization,
    )


_SIGNATURE, _PARAMETRIZATION = create_parametrization(
    PLOTS,
    NAMED_SCENARIOS,
    FAST_FLAG,
    OUTCOMES,
)


@pytask.mark.depends_on(_MODULE_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_plot_scenario_comparison(
    depends_on,
    outcome,
    title,
    colors,
    name_to_label,
    scenario_starts,
    plot_start,
    empirical,
    produces,
):
    """Plot comparisons between the incidences of several scenarios.

    Args:
        depends_on (dict): keys contain py files and scenario names. Values are
            paths to the dependencies.
        outcome (str): name of the incidence to be plotted (new_known_case or
            newly_infected).
        title (str): custom title, will be formatted with New Observed Cases or
            Total New Cases depending on the outcome.
        colors (dict, optional): keys are scenario names, values are colors.
        name_to_label (dict, optional): keys are scenario names, values are the
            labels to be put in the legend. If None they are generated automatically
            from the scenario names.
        plot_start (pandas.Timestamp, optional): date on which the plot should start.
            If None, the plot start is the simulation start.
        empirical (bool, optional): whether to plot empirical equivalents.
            If not given, they are plotted if an empirical analogue for the outcome is
            available.
        produces (pathlib.Path): path where to save the figure

    """
    # drop py file dependencies
    depends_on = filter_dictionary(
        lambda x: not x.endswith(".py") and "cosmo" not in x, depends_on
    )

    # prepare the plot inputs
    dfs = {name: pd.read_pickle(path) for name, path in depends_on.items()}
    dfs = shorten_dfs(dfs, empirical=empirical, plot_start=plot_start)

    title = _create_title(title, outcome)
    name_to_label = (
        create_automatic_labels(dfs) if name_to_label is None else name_to_label
    )

    missing_labels = [x for x in dfs.keys() if x not in name_to_label]
    assert (
        len(missing_labels) == 0
    ), f"You did not specify a label for {missing_labels}."

    empirical_available = [
        "new_known_case",
        "newly_deceased",
        "share_ever_rapid_test",
        "share_rapid_test_in_last_week",
    ]
    if empirical is None:
        empirical = outcome if outcome in empirical_available else False

    # create the plots
    fig, ax = plot_incidences(
        incidences=dfs,
        title=title,
        name_to_label=name_to_label,
        empirical=empirical,
        colors=colors,
        scenario_starts=scenario_starts,
    )
    plt.savefig(produces["fig"], dpi=200, transparent=False, facecolor="w")
    plt.close()

    # save data for report write up as .csv
    weekly_mean_values = pd.DataFrame()
    for key, df in dfs.items():
        mean_over_seeds = df.mean(axis=1)
        mean_over_weeks = mean_over_seeds.groupby(pd.Grouper(freq="W")).mean()
        weekly_mean_values[key] = mean_over_weeks.round(2)

    weekly_mean_values.to_csv(produces["data"])


def _create_title(title, outcome):
    name_to_nice_name = {
        "new_known_case": "Observed New Cases",
        "newly_infected": "Total New Cases",
        "newly_deceased": "New Deaths",
        "share_ever_rapid_test": "Share of People who Have Ever Done a Rapid Test\n",
        "share_rapid_test_in_last_week": "Share of People who Have Done a Rapid Test\n"
        + "in the Last Week",
        "r_effective": "the Effective Reproduction Number",
    }

    title_outcome = name_to_nice_name.get(outcome, outcome.replace("_", " ").title())
    title = title.format(outcome=title_outcome)
    return title
