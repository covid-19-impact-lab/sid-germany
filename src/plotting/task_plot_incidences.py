import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.plotting.plotting import plot_incidences
from src.plotting.plotting import PY_DEPENDENCIES
from src.policies.policy_tools import filter_dictionary
from src.simulation.scenario_config import (
    create_path_to_weekly_outcome_of_scenario,
)
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios
from src.simulation.task_run_simulation import FAST_FLAG

NAMED_SCENARIOS = get_named_scenarios()


PLOTS = {
    "fall": ["fall_baseline"],
    "effect_of_vaccines": ["summer_baseline", "spring_without_vaccines"],
    "effect_of_rapid_tests": [
        "summer_baseline",
        "spring_without_rapid_tests_at_schools",
        "spring_without_rapid_tests_at_work",
        "spring_without_rapid_tests",
        "spring_with_obligatory_work_rapid_tests",
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
        "spring_emergency_care_after_easter_no_school_rapid_tests",
        "spring_educ_open_after_easter",
        "spring_educ_open_after_easter_educ_tests_every_other_day",
        "spring_educ_open_after_easter_educ_tests_every_day",
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
assert AVAILABLE_SCENARIOS.issubset(
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
            to_compare = available_scenarios.intersection(to_compare)
            depends_on = {
                scenario_name: create_path_to_weekly_outcome_of_scenario(
                    name=scenario_name, outcome=outcome, groupby=None
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


@pytask.mark.depends_on(PY_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_plot_weekly_outcomes(depends_on, comparison_name, outcome, produces):
    # drop py file dependencies
    depends_on = filter_dictionary(lambda x: not x.startswith("py_"), depends_on)

    df_dict = {name: pd.read_pickle(path) for name, path in depends_on.items()}
    nice_name = comparison_name.replace("_", " ").title()
    title = f"{outcome.replace('_', ' ').title()} in {nice_name}"

    name_to_label = _create_nice_labels(df_dict)

    fig, ax = plot_incidences(
        incidences=df_dict,
        title=title,
        name_to_label=name_to_label,
        rki=outcome == "new_known_case",
        plot_scenario_start="summer" in comparison_name,
    )
    plt.savefig(produces, dpi=200, transparent=False, facecolor="w")
    plt.close()


def _create_nice_labels(df_dict):
    name_to_label = {}
    replacements = [("fall", ""), ("spring", ""), ("summer", ""), ("_", " ")]
    for name in df_dict:
        nice_name = name
        for old, new in replacements:
            nice_name = nice_name.replace(old, new)
        name_to_label[name] = nice_name.title()
    return name_to_label
