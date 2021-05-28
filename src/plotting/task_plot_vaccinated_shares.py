import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import SRC
from src.plotting.plotting import create_nice_labels
from src.plotting.plotting import make_name_nice
from src.plotting.plotting import plot_group_time_series
from src.plotting.plotting import plot_incidences
from src.plotting.plotting import shorten_dfs
from src.policies.policy_tools import combine_dictionaries
from src.simulation.scenario_config import create_path_to_scenario_outcome_time_series
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios


NAMED_SCENARIOS = get_named_scenarios()
AVAILABLE_SCENARIOS = get_available_scenarios(NAMED_SCENARIOS)
VACCINATION_SCENARIOS = [
    name
    for name in AVAILABLE_SCENARIOS
    if ("vaccin" in name) or (name == "spring_baseline")
]

_JOINT_DEPENDENCIES = {
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
    "plotting.py": SRC / "plotting" / "plotting.py",
    "actual_vaccinations": BLD
    / "data"
    / "vaccinations"
    / "vaccination_shares_extended.pkl",
}


def _create_comparison_dependencies():
    simulation_dependencies = {
        name: create_path_to_scenario_outcome_time_series(name, "ever_vaccinated")
        for name in VACCINATION_SCENARIOS
    }
    dependencies = combine_dictionaries([_JOINT_DEPENDENCIES, simulation_dependencies])
    return dependencies


_COMPARISON_DEPENDENCIES = _create_comparison_dependencies()


@pytask.mark.depends_on(_COMPARISON_DEPENDENCIES)
@pytask.mark.produces(
    BLD / "figures" / "vaccinations" / "comparison_across_scenarios.png"
)
def task_plot_overall_vaccination_shares_across_scenarios(depends_on, produces):
    dfs = {
        name: pd.read_pickle(path)
        for name, path in depends_on.items()
        if name in VACCINATION_SCENARIOS
    }
    dfs = shorten_dfs(dfs)
    title = "Comparison of Vaccination Rates Across Scenarios"

    name_to_label = create_nice_labels(dfs)

    fig, ax = plot_incidences(
        incidences=dfs,
        title=title,
        name_to_label=name_to_label,
        rki=False,
        colors=None,
        scenario_starts=None,
    )

    # add the actual vaccination shares
    actual_vacc_shares = pd.read_pickle(depends_on["actual_vaccinations"]).cumsum()
    dates = list(dfs.values())[0].index
    start = dates.min()
    end = dates.max()
    actual_vacc_shares = actual_vacc_shares[start:end]
    sns.lineplot(
        x=actual_vacc_shares.index,
        y=actual_vacc_shares,
        label="actual share of vaccinated people",
        ax=ax,
    )
    ax.set_ylabel("share of vaccinated individuals")

    plt.savefig(produces, dpi=200, transparent=False, facecolor="w")
    plt.close()


_PARAMETRIZATION = []
for name in VACCINATION_SCENARIOS:
    dep = {
        "simulated": create_path_to_scenario_outcome_time_series(
            name, "ever_vaccinated_by_age_group_rki"
        ),
    }
    produces = BLD / "figures" / "vaccinations" / f"{name}.png"
    _PARAMETRIZATION.append((name, dep, produces))


@pytask.mark.depends_on(_JOINT_DEPENDENCIES)
@pytask.mark.parametrize("name, depends_on, produces", _PARAMETRIZATION)
def task_plot_groupby_vaccination_shares(name, depends_on, produces):
    vaccination_shares = pd.read_pickle(depends_on["simulated"])

    nice_name = make_name_nice(name)
    title = "Share of Vaccinated People by Age Group {group} in " + nice_name.title()
    fig, axes = plot_group_time_series(df=vaccination_shares, title=title, rki=None)
    for ax in axes:
        ax.set_ylabel("share of vaccinated individuals")

    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")
    plt.close()