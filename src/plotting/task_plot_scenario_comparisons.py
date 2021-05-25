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
    "fall": {
        "title": "{outcome} in Fall",
        "scenarios": ["fall_baseline"],
        "scenario_starts": None,
        "colors": [BLUE],
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
        "colors": [BLUE, BROWN, RED, ORANGE],
        "scenario_starts": None,
        "plot_start": None,
    },
    "explaining_the_decline": {
        "title": "Explaining the Puzzling Decline in {outcome}",
        "scenarios": [
            "spring_baseline",
            "spring_without_vaccines",
            "spring_without_rapid_tests",
            "spring_without_rapid_tests_and_no_vaccinations",
            "spring_without_seasonality",
            "spring_without_rapid_tests_without_vaccinations_without_seasonality",
        ],
        "scenario_starts": None,
        "colors": None,
        "plot_start": None,
    },
    "school_scenarios": {
        "title": "The Effect of Schools on {outcome}",
        "scenarios": [
            "spring_educ_open_after_easter_without_tests",
            "spring_educ_open_after_easter_with_normal_tests",
            "spring_baseline",
            "spring_emergency_care_after_easter_without_school_rapid_tests",
        ],
        "scenario_starts": None,
        "colors": [RED, YELLOW, BLUE, GREEN],
        "plot_start": AFTER_EASTER,
    },
    "vaccine_scenarios": {
        "title": "Effect of Different Vaccination Scenarios on {outcome}",
        "scenarios": [
            "spring_baseline",
            "spring_vaccinate_1_pct_per_day_after_easter",
            "spring_without_vaccines",
        ],
        "scenario_starts": ([(AFTER_EASTER, "start of increased vaccinations")]),
        "colors": [BLUE, GREEN, RED],
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
                scenario_name: create_path_to_weekly_outcome_of_scenario(
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
                        plot_info["scenario_starts"],
                        plot_info["plot_start"],
                        produces,
                    )
                )

    return (
        "depends_on, outcome, title, colors, scenario_starts, plot_start, produces",
        parametrization,
    )


_SIGNATURE, _PARAMETRIZATION = create_parametrization(
    PLOTS, NAMED_SCENARIOS, FAST_FLAG, ["newly_infected", "new_known_case"]
)


@pytask.mark.depends_on(_MODULE_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_plot_scenario_comparison(
    depends_on, outcome, title, colors, scenario_starts, plot_start, produces
):
    # drop py file dependencies
    depends_on = filter_dictionary(lambda x: not x.endswith(".py"), depends_on)

    dfs = {name: pd.read_pickle(path) for name, path in depends_on.items()}
    dfs = _shorten_dfs(dfs, plot_start)

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


def _shorten_dfs(dfs, plot_start):
    """Shorten all incidence DataFrames.

    All DataFrames are shortened to the shortest. In addition, if plot_start is given
    all DataFrames start at or after plot_start.

    Args:
        dfs (dict): keys are the names of the scenarios, values are the incidence
            DataFrames.
        plot_start (pd.Timestamp or None): earliest allowed start date for the plot

    Returns:
        shortened (dict): keys are the names of the scenarios, values are the shortened
            DataFrames.

    """
    shortened = {}

    start_date = max(df.index.min() for df in dfs.values())
    if plot_start is not None:
        start_date = max(plot_start, start_date)
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
