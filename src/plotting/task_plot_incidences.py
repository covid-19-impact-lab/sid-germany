import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.plotting.plotting import plot_incidences
from src.policies.policy_tools import filter_dictionary
from src.simulation.task_process_simulation_outputs import (
    create_path_for_weekly_outcome_of_scenario,
)
from src.simulation.task_run_simulation import FAST_FLAG
from src.simulation.task_run_simulation import NAMED_SCENARIOS

PY_DEPENDENCIES = {
    "py_config": SRC / "config.py",
    "py_plot_incidences": SRC / "plotting" / "plotting.py",
    "py_process_sim_outputs": SRC / "simulation" / "task_process_simulation_outputs.py",
}

PLOTS = {
    "fall": ["fall_baseline"],
    "effect_of_vaccines": ["spring_baseline", "spring_without_vaccines"],
    "effect_of_rapid_tests": [
        "spring_baseline",
        "spring_without_rapid_tests_at_schools",
        "spring_without_rapid_tests_at_work",
        "spring_without_rapid_tests",
        "spring_with_obligatory_work_rapid_tests",
    ],
    "vaccines_vs_rapid_tests": [
        "spring_baseline",
        "spring_without_vaccines",
        # maybe replace the rapid test scenario with a different one.
        "spring_without_rapid_tests",
        "spring_with_more_vaccines",
    ],
    "rapid_tests_vs_school_closures": [
        "spring_baseline",
        "spring_emergency_care_after_easter_no_school_rapid_tests",
        "spring_educ_open_after_easter",
        "spring_educ_open_after_easter_educ_tests_every_other_day",
        "spring_educ_open_after_easter_educ_tests_every_day",
    ],
    "future": [
        "future_baseline",
        "future_educ_open",
        "future_reduced_test_demand",
        "future_strict_home_office",
        "future_optimistic_vaccinations",
        "future_more_rapid_tests_at_work",
    ],
}
"""Dict[str, List[str]]: A dictionary containing the plots to create.

Each key in the dictionary is a name for a collection of scenarios. The values are lists
of scenario names which are combined to create the collection.

"""

available_scenarios = {
    name for name, spec in NAMED_SCENARIOS.items() if spec["n_seeds"] > 0
}
plotted_scenarios = {x for scenarios in PLOTS.values() for x in scenarios}
assert available_scenarios.issubset(
    plotted_scenarios
), "The following scenarios do not appear in any plots: " + "\n\t".join(
    available_scenarios.difference(plotted_scenarios)
)


def create_path_for_figure_of_weekly_outcome_of_scenario(name, fast_flag, outcome):
    return BLD / "figures" / f"{fast_flag}_{name}_{outcome}.png"


def create_parametrization(plots, named_scenarios, fast_flag, outcomes):
    available_scenarios = {
        name for name, spec in NAMED_SCENARIOS.items() if spec["n_seeds"] > 0
    }
    parametrization = []
    for outcome in outcomes:
        for comparison_name, to_compare in plots.items():
            to_compare = available_scenarios.intersection(to_compare)
            depends_on = {
                scenario_name: create_path_for_weekly_outcome_of_scenario(
                    scenario_name, fast_flag, outcome, None
                )
                for scenario_name in to_compare
            }

            missing_scenarios = set(depends_on) - set(named_scenarios)
            if missing_scenarios:
                raise ValueError(f"Some scenarios are missing: {missing_scenarios}.")

            produces = create_path_for_figure_of_weekly_outcome_of_scenario(
                comparison_name, fast_flag, outcome
            )
            parametrization.append((depends_on, comparison_name, outcome, produces))

    return "depends_on, comparison_name, outcome, produces", parametrization


SIGNATURE, PARAMETRIZATION = create_parametrization(
    PLOTS, NAMED_SCENARIOS, FAST_FLAG, ["newly_infected", "new_known_case"]
)


@pytask.mark.depends_on(PY_DEPENDENCIES)
@pytask.mark.parametrize(SIGNATURE, PARAMETRIZATION)
def task_plot_weekly_outcomes(depends_on, comparison_name, outcome, produces):
    # drop py file dependencies
    depends_on = filter_dictionary(lambda x: not x.startswith("py_"), depends_on)

    dfs = {name: pd.read_pickle(path) for name, path in depends_on.items()}
    nice_name = comparison_name.replace("_", " ").title()
    title = f"{outcome.replace('_', ' ').title()} in {nice_name}"

    fig, ax = plot_incidences(
        incidences=dfs,
        title=title,
        name_to_label={name: name.replace("_", " ") for name in dfs},
        rki=outcome,
        plot_scenario_start="future" in comparison_name,
    )
    plt.savefig(produces, dpi=200, transparent=False, facecolor="w")
