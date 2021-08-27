import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import FAST_FLAG
from src.config import SRC
from src.plotting.plotting import BLUE
from src.plotting.plotting import create_automatic_labels
from src.plotting.plotting import make_nice_outcome
from src.plotting.plotting import OUTCOME_TO_EMPIRICAL_LABEL
from src.plotting.plotting import OUTCOME_TO_Y_LABEL
from src.plotting.plotting import plot_incidences
from src.plotting.plotting import shorten_dfs
from src.policies.policy_tools import filter_dictionary
from src.simulation.scenario_config import create_path_for_weekly_outcome_of_scenario
from src.simulation.scenario_config import create_path_to_scenario_outcome_time_series
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios


_MODULE_DEPENDENCIES = {
    "plotting.py": SRC / "plotting" / "plotting.py",
    "policy_tools.py": SRC / "policies" / "policy_tools.py",
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
    "empirical": BLD / "data" / "empirical_data_for_plotting.pkl",
}

NAMED_SCENARIOS = get_named_scenarios()


OUTCOMES = [
    "newly_infected",
    "new_known_case",
    "newly_deceased",
    "share_ever_rapid_test",
    "share_rapid_test_in_last_week",
    "share_b117",
    "share_delta",
    "share_doing_rapid_test_today",
    "ever_vaccinated",
]

if FAST_FLAG != "debug":
    OUTCOMES.append("r_effective")


PLOTS = {
    "summer_baseline": {
        "title": "",
        "scenarios": ["summer_baseline"],
        "colors": [BLUE],
        "name_to_label": {"summer_baseline": "summer and fall with open schools"},
        "empirical": True,
    },
    # Main Plots (Fixed)
    "combined_fit": {
        "title": "Simulated versus Empirical: {outcome}",
        "scenarios": ["combined_baseline"],
        "name_to_label": {"combined_baseline": "simulated"},
        "colors": [BLUE],
        "empirical": True,
    },
}
"""Dict[str, Dict[str, str]]: A dictionary containing the plots to create.

Each key in the dictionary is a name for a collection of scenarios. The values are
dictionaries with the title and the lists of scenario names which are combined to
create the collection.

"""

AVAILABLE_SCENARIOS = get_available_scenarios(NAMED_SCENARIOS)

plotted_scenarios = {x for spec in PLOTS.values() for x in spec["scenarios"]}
not_plotted_scenarios = set(AVAILABLE_SCENARIOS) - plotted_scenarios
# Remove scenarios which are not directly used in plots.
not_plotted_scenarios = not_plotted_scenarios - {
    "spring_with_only_vaccinations",
    "spring_with_only_rapid_tests",
    "fall_baseline",
    "spring_baseline",
}
if not_plotted_scenarios:
    raise ValueError(
        "The following scenarios do not appear in any plots: "
        + "\n\t".join(not_plotted_scenarios)
    )


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
                assert (
                    name in named_scenarios
                ), f"The scenario {name} is not a scenario that is ever run. Typo?"

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
                    comparison_name, fast_flag, outcome, "pdf"
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
                        plot_info.get("empirical", False),
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
        outcome (str): name of the incidence to be plotted (e.g. new_known_case or
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
        produces (pathlib.Path): path where to save the figure

    """
    empirical_df = pd.read_pickle(depends_on["empirical"])
    # drop py file dependencies
    depends_on = filter_dictionary(
        lambda x: not x.endswith(".py") and "empirical" not in x, depends_on
    )

    # prepare the plot inputs
    dfs = {name: pd.read_pickle(path) for name, path in depends_on.items()}

    fig_path = produces["fig"]
    nice_outcome = make_nice_outcome(outcome)
    title = title.format(outcome=nice_outcome)
    name_to_label = (
        create_automatic_labels(dfs) if name_to_label is None else name_to_label
    )

    first_df = list(dfs.values())[0]
    # save the actual x limits before dfs are shortened
    if plot_start is None:
        xlims = first_df.index.min(), first_df.index.max()
    else:
        xlims = plot_start, first_df.index.max()

    plot_end = pd.Timestamp("2021-05-01") if outcome == "share_b117" else None
    dfs = shorten_dfs(dfs, plot_start=plot_start, plot_end=plot_end)
    dates = list(dfs.values())[0].index

    if empirical and outcome in empirical_df.columns:
        dfs["empirical"] = empirical_df.loc[dates.min() : dates.max(), [outcome]]
        name_to_label["empirical"] = OUTCOME_TO_EMPIRICAL_LABEL[outcome]

    missing_labels = [x for x in dfs.keys() if x not in name_to_label]
    assert (
        len(missing_labels) == 0
    ), f"You did not specify a label for {missing_labels}."

    # create the plots
    fig, ax = plot_incidences(
        incidences=dfs,
        title="",  # Do not use the title for the paper version
        name_to_label=name_to_label,
        colors=colors,
        scenario_starts=scenario_starts,
        n_single_runs=0,
        ylabel=OUTCOME_TO_Y_LABEL.get(outcome, None),
    )
    ax.set_xlim(xlims)

    if "new_work_scenarios" in str(produces):
        x, y, width, height = 0.0, -0.3, 1, 0.2
        ax.legend(loc="upper center", bbox_to_anchor=(x, y, width, height), ncol=3)
    fig.tight_layout()
    fig.savefig(fig_path)

    # save with single run lines
    with_single_runs_path = fig_path.parent / (
        fig_path.stem + "_with_single_runs" + fig_path.suffix
    )
    fig_with_lines, ax_with_lines = plot_incidences(
        incidences=dfs,
        title="",  # Do not use the title for the paper version
        name_to_label=name_to_label,
        colors=colors,
        scenario_starts=scenario_starts,
        n_single_runs=None,
        ylabel=OUTCOME_TO_Y_LABEL.get(outcome, None),
    )
    ax_with_lines.set_xlim(xlims)
    if "new_work_scenarios" in str(produces):
        x, y, width, height = 0.0, -0.3, 1, 0.2
        ax_with_lines.legend(
            loc="upper center", bbox_to_anchor=(x, y, width, height), ncol=3
        )

    fig_with_lines.savefig(with_single_runs_path)

    # crop if necessary
    min_y, max_y = ax.get_ylim()
    cropped_path = fig_path.parent / (fig_path.stem + "_cropped" + fig_path.suffix)
    if outcome == "new_known_case" and max_y > 510:
        ax.set_ylim(min_y, 510)
        fig.savefig(cropped_path)
    elif outcome == "newly_infected" and max_y > 1050:
        ax.set_ylim(min_y, 1050)
        fig.savefig(cropped_path)
    elif outcome == "newly_deceased" and max_y > 20.5:
        ax.set_ylim(min_y, 20.5)
        fig.savefig(cropped_path)
    plt.close()

    # save data for report write up as .csv
    plot_data = pd.DataFrame()
    for key, df in dfs.items():
        plot_data[key] = df.mean(axis=1)
    plot_data.to_csv(produces["data"])
