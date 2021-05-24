import matplotlib.pyplot as plt
import pandas as pd
import pytask
import sid

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


PLOTS = {
    "fall": ["fall_baseline"],
    "effect_of_vaccines": [
        "summer_baseline",
        "spring_without_vaccines",
        "spring_with_more_vaccines",
    ],
    "effect_of_rapid_tests": [
        "summer_baseline",
        "spring_without_school_rapid_tests",
        "spring_without_work_rapid_tests",
        "spring_without_rapid_tests",
        "spring_with_mandatory_work_rapid_tests",
    ],
    "vaccines_vs_rapid_tests": [
        "summer_baseline",
        "spring_without_vaccines",
        # maybe replace the rapid test scenario with a different one.
        "spring_without_rapid_tests",
        "spring_with_more_vaccines",
    ],
    "rapid_tests_vs_school_closures": [
        "summer_baseline",
        "spring_emergency_care_after_easter_without_school_rapid_tests",
        "spring_educ_open_after_easter",
        "spring_open_educ_after_easter_with_tests_every_other_day",
        "spring_open_educ_after_easter_with_daily_tests",
    ],
    "summer": [
        "summer_baseline",
        "summer_educ_open",
        "summer_reduced_test_demand",
        "summer_strict_home_office",
        "summer_optimistic_vaccinations",
        "summer_more_rapid_tests_at_work",
    ],
}
"""Dict[str, List[str]]: A dictionary containing the plots to create.

Each key in the dictionary is a name for a collection of scenarios. The values are lists
of scenario names which are combined to create the collection.

"""

AVAILABLE_SCENARIOS = get_available_scenarios(NAMED_SCENARIOS)

plotted_scenarios = {x for scenarios in PLOTS.values() for x in scenarios}
assert set(AVAILABLE_SCENARIOS).issubset(
    plotted_scenarios
), "The following scenarios do not appear in any plots: " + "\n\t".join(
    AVAILABLE_SCENARIOS.difference(plotted_scenarios)
)


def create_path_for_figure_of_weekly_outcome_of_scenario(name, fast_flag, outcome):
    return BLD / "figures" / f"{fast_flag}_{name}_{outcome}.png"


def create_parametrization(plots, named_scenarios, fast_flag, outcomes):
    available_scenarios = get_available_scenarios(named_scenarios)
    parametrization = []
    for outcome in outcomes:
        for comparison_name, to_compare in plots.items():
            to_compare = sorted(set(available_scenarios).intersection(to_compare))
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
                parametrization.append((depends_on, comparison_name, outcome, produces))

    return "depends_on, comparison_name, outcome, produces", parametrization


_SIGNATURE, _PARAMETRIZATION = create_parametrization(
    PLOTS, NAMED_SCENARIOS, FAST_FLAG, ["newly_infected", "new_known_case"]
)


@pytask.mark.depends_on(_MODULE_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_plot_scenario_comparison(depends_on, comparison_name, outcome, produces):
    # drop py file dependencies
    depends_on = filter_dictionary(lambda x: not x.endswith(".py"), depends_on)

    dfs = {name: pd.read_pickle(path) for name, path in depends_on.items()}
    dfs = _shorten_dfs_to_the_shortest(dfs)

    title = _create_title(comparison_name, outcome)
    name_to_label = _create_nice_labels(dfs)

    colors = sid.get_colors("categorical", len(dfs))
    # 3rd entry is not well distinguishable from the first
    if len(colors) >= 3:
        colors[2] = "#2E8B57"  # seagreen

    fig, ax = plot_incidences(
        incidences=dfs,
        title=title,
        colors=colors,
        name_to_label=name_to_label,
        rki=outcome == "new_known_case",
        plot_scenario_start="summer" in comparison_name,
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


def _create_title(comparison_name, outcome):
    nice_name = comparison_name.replace("_", " ").title()
    if outcome == "new_known_case":
        title_outcome = "Observed New Cases"
    elif outcome == "newly_infected":
        title_outcome = "Total New Cases"
    title = f"{title_outcome} in {nice_name}"
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
