import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import FAST_FLAG
from src.config import SRC
from src.plotting.plotting import plot_incidences
from src.policies.policy_tools import filter_dictionary
from src.simulation.scenario_config import create_path_to_weekly_outcome_of_scenario
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios

_MODULE_DEPENDENCIES = {
    "plotting.py": SRC / "plotting" / "plotting.py",
    "policy_tools.py": SRC / "policies" / "policy_tools.py",
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
}

NAMED_SCENARIOS = get_named_scenarios()

SID_BLUE = "#547482"

PLOTS = {
    "fall": {
        "title": "{outcome} in Fall",
        "scenarios": ["fall_baseline"],
        "scenario_starts": None,
        "colors": [SID_BLUE],
    },
    "effect_of_vaccines": {
        "title": "{outcome} in Different Vaccine Scenarios",
        "scenarios": [
            "summer_baseline",
            "spring_without_vaccines",
            "spring_with_more_vaccines",
        ],
        "scenario_starts": [
            (pd.Timestamp("2021-02-15"), "vaccine stop"),
            (pd.Timestamp("2021-04-06"), "start increased vaccinations"),
        ],
        "colors": [SID_BLUE, "steelblue", "dodgerblue"],
    },
    "effect_of_rapid_tests": {
        "title": "{outcome} in Different Rapid Test Scenarios",
        "scenarios": [
            "summer_baseline",
            "spring_without_rapid_tests",
            "spring_without_work_rapid_tests",
            "spring_without_school_rapid_tests",
            "spring_with_mandatory_work_rapid_tests",
        ],
        "scenario_starts": None,
        "colors": None,
    },
    "vaccines_vs_rapid_tests": {
        "title": "The Effect of Vaccines on {outcome} Compared to Rapid Tests",
        "scenarios": [
            "summer_baseline",
            # vaccines
            "spring_without_vaccines",
            "spring_with_more_vaccines",
            # rapid tests
            "spring_without_rapid_tests",
            "spring_with_mandatory_work_rapid_tests",
        ],
        "scenario_starts": None,
        "colors": None,
    },
    "school_scenarios": {
        "title": "{outcome} in Different School Scenarios",
        "scenarios": [
            "summer_baseline",
            "spring_emergency_care_after_easter_without_school_rapid_tests",
            # different test scenarios
            "spring_educ_open_after_easter_without_tests",
            "spring_educ_open_after_easter_with_normal_tests",
            "spring_open_educ_after_easter_with_tests_every_other_day",
            "spring_open_educ_after_easter_with_daily_tests",
        ],
        "scenario_starts": None,
        "colors": None,
    },
    "summer": {
        "title": "Summer Prediction for {outcome}",
        "scenarios": [
            "summer_baseline",
            "summer_educ_open",
            "summer_reduced_test_demand",
            "summer_strict_home_office",
            "summer_optimistic_vaccinations",
            "summer_more_rapid_tests_at_work",
        ],
        "scenario_starts": None,
        "colors": None,
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


def create_path_for_figure_of_weekly_outcome_of_scenario(name, fast_flag, outcome):
    return BLD / "figures" / f"{fast_flag}_{name}_{outcome}.png"


def create_parametrization(plots, named_scenarios, fast_flag, outcomes):
    available_scenarios = get_available_scenarios(named_scenarios)
    parametrization = []
    for outcome in outcomes:
        for comparison_name, plot_info in plots.items():
            title = plot_info["title"]
            to_compare = sorted(
                set(available_scenarios).intersection(plot_info["scenarios"])
            )
            depends_on = {
                scenario_name: create_path_to_weekly_outcome_of_scenario(
                    name=scenario_name, entry=outcome
                )
                for scenario_name in to_compare
            }

            missing_scenarios = set(depends_on) - set(named_scenarios)
            if missing_scenarios:
                raise ValueError(f"Some scenarios are missing: {missing_scenarios}.")

            produces = create_path_for_figure_of_weekly_outcome_of_scenario(
                comparison_name, fast_flag, outcome
            )
            # only create a plot if at least one scenario had a seed.
            if depends_on:
                parametrization.append(
                    (
                        depends_on,
                        outcome,
                        title,
                        plot_info["colors"],
                        plot_info["scenario_starts"],
                        produces,
                    )
                )

    return (
        "depends_on, outcome, title, colors, scenario_starts, produces",
        parametrization,
    )


_SIGNATURE, _PARAMETRIZATION = create_parametrization(
    PLOTS, NAMED_SCENARIOS, FAST_FLAG, ["newly_infected", "new_known_case"]
)


@pytask.mark.depends_on(_MODULE_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_plot_scenario_comparison(
    depends_on, outcome, title, colors, scenario_starts, produces
):
    # drop py file dependencies
    depends_on = filter_dictionary(lambda x: not x.endswith(".py"), depends_on)

    dfs = {name: pd.read_pickle(path) for name, path in depends_on.items()}
    dfs = _shorten_dfs_to_the_shortest(dfs)

    title = _create_title(title, outcome)
    name_to_label = _create_nice_labels(dfs)

    fig, ax = plot_incidences(
        incidences=dfs,
        title=title,
        name_to_label=name_to_label,
        rki=outcome == "new_known_case",
        colors=colors,
        scenario_starts=scenario_starts,
    )
    plt.savefig(produces, dpi=200, transparent=False, facecolor="w")
    plt.close()


def _shorten_dfs_to_the_shortest(dfs):
    """Shorten all incidence DataFrames to the time frame of the shortest.

    Args:
        dfs (dict): keys are the names of the scenarios, values are the incidence
            DataFrames.

    Returns:
        shortened (dict): keys are the names of the scenarios, values are the shortened
            DataFrames.

    """
    shortened = {}

    start_date = max(df.index.min() for df in dfs.values())
    end_date = min(df.index.max() for df in dfs.values())

    for name, df in dfs.items():
        shortened[name] = df.loc[start_date:end_date].copy(deep=True)

    return shortened


def _create_title(title, outcome):
    if outcome == "new_known_case":
        title_outcome = "Observed New Cases"
    elif outcome == "newly_infected":
        title_outcome = "Total New Cases"
    title = title.format(outcome=title_outcome)
    return title


def _create_nice_labels(dfs):
    name_to_label = {}
    replacements = [
        ("_", " "),
        (" with", "\n with"),
        ("fall", ""),
        ("spring", ""),
        ("summer", ""),
    ]
    for name in dfs:
        nice_name = name
        for old, new in replacements:
            nice_name = nice_name.replace(old, new)
        name_to_label[name] = nice_name.lstrip("\n")
    return name_to_label
