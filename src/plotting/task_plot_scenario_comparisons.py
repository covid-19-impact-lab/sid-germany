import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import FAST_FLAG
from src.config import SRC
from src.plotting.plotting import create_nice_labels
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
}

NAMED_SCENARIOS = get_named_scenarios()

AFTER_EASTER = pd.Timestamp("2021-04-06")

# Colors
BLUE = "#4e79a7"
ORANGE = "#f28e2b"
RED = "#e15759"
TEAL = "#76b7b2"
GREEN = "#59a14f"
YELLOW = "#edc948"
PURPLE = "#b07aa1"
BROWN = "#9c755f"


PLOTS = {
    # Fixed Plots
    "fall": {
        "title": "{outcome} in Fall",
        "scenarios": ["fall_baseline"],
        "name_to_label": {"fall_baseline": "simulation"},
        "colors": [BLUE],
        "scenario_starts": None,
        "plot_start": None,
    },
    "effect_of_rapid_tests": {
        "title": "Decomposing the Effect of Rapid Tests on {outcome}",
        "scenarios": [
            "spring_baseline",
            "spring_without_rapid_tests",
            "spring_without_work_rapid_tests",
            "spring_without_school_rapid_tests",
        ],
        "name_to_label": {
            "spring_baseline": "with all effects",
            "spring_without_rapid_tests": "without any rapid tests",
            "spring_without_work_rapid_tests": "without work rapid tests",
            "spring_without_school_rapid_tests": "without school rapid tests",
        },
        "colors": [BLUE, BROWN, RED, ORANGE],
        "scenario_starts": None,
        "plot_start": None,
    },
    "explaining_the_decline": {
        "title": "Explaining the Puzzling Decline in {outcome}",
        "scenarios": [
            "spring_baseline",
            "spring_without_rapid_tests",
            "spring_without_rapid_tests_and_no_vaccinations",
            "spring_no_effects",
        ],
        "name_to_label": {
            "spring_baseline": "with all channels",
            "spring_without_rapid_tests": "with vaccinations and seasonality",
            "spring_without_rapid_tests_and_no_vaccinations": "with seasonality only",
            "spring_no_effects": "without any channel",
        },
        "colors": None,
        "scenario_starts": None,
        "plot_start": None,
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
        "scenario_starts": None,
        "plot_start": None,
    },
    # Variable Plots
    "school_scenarios": {
        "title": "The Effect of Schools on {outcome}",
        "scenarios": [
            "spring_educ_open_after_easter_without_tests",
            "spring_educ_open_after_easter_with_tests",
            "spring_baseline",
            "spring_close_educ_after_easter",
        ],
        "name_to_label": {
            "spring_educ_open_after_easter_without_tests": "open schools without tests",
            "spring_educ_open_after_easter_with_tests": "open schools with tests",
            "spring_baseline": "current school and test policy",
            "spring_close_educ_after_easter": "keep schools closed",
        },
        "colors": [RED, YELLOW, BLUE, GREEN],
        "scenario_starts": [(AFTER_EASTER, "Easter")],
        "plot_start": AFTER_EASTER,
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
            "of the population\n every day after Easter",
            "spring_without_vaccines": "no vaccinations after February 10th",
        },
        "colors": [BLUE, GREEN, RED],
        "scenario_starts": ([(AFTER_EASTER, "start of increased vaccinations")]),
        "plot_start": None,
    },
    "illustrate_rapid_tests": {
        "title": "Illustrate the effect of rapid tests on {outcome}",
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
        "plot_start": None,
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
                    name=scenario_name, entry=outcome
                )
                for scenario_name in scenarios
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
                        colors,
                        plot_info["name_to_label"],
                        plot_info["scenario_starts"],
                        plot_info["plot_start"],
                        produces,
                    )
                )

    return (
        "depends_on, outcome, title, colors, name_to_label, scenario_starts, "
        "plot_start, produces",
        parametrization,
    )


_SIGNATURE, _PARAMETRIZATION = create_parametrization(
    PLOTS, NAMED_SCENARIOS, FAST_FLAG, ["newly_infected", "new_known_case"]
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
    produces,
):
    # drop py file dependencies
    depends_on = filter_dictionary(lambda x: not x.endswith(".py"), depends_on)

    dfs = {name: pd.read_pickle(path) for name, path in depends_on.items()}
    dfs = shorten_dfs(dfs, plot_start)

    title = _create_title(title, outcome)
    name_to_label = create_nice_labels(dfs) if name_to_label is None else name_to_label

    missing_labels = [x for x in dfs.keys() if x not in name_to_label]
    assert (
        len(missing_labels) == 0
    ), f"You did not specify a label for {missing_labels}."

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


def _create_title(title, outcome):
    if outcome == "new_known_case":
        title_outcome = "Observed New Cases"
    elif outcome == "newly_infected":
        title_outcome = "Total New Cases"
    title = title.format(outcome=title_outcome)
    return title
