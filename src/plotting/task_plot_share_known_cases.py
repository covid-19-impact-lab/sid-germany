"""Create plots, illustrating the share known cases over time."""
import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import SRC
from src.plotting.plotting import make_scenario_name_nice
from src.plotting.plotting import plot_share_known_cases
from src.simulation.scenario_config import (
    create_path_to_share_known_cases_of_scenario,
)
from src.simulation.scenario_config import create_path_to_share_known_cases_plot
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios

_MODULE_DEPENDENCIES = {
    "plotting.py": SRC / "plotting" / "plotting.py",
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
}


def _create_parametrization():
    """Create the parametrization for the share known cases plots."""
    named_scenarios = get_named_scenarios()
    available_scenarios = get_available_scenarios(named_scenarios)
    parametrization = []
    for scenario_name in available_scenarios:
        nice_name = make_scenario_name_nice(scenario_name).title()
        for groupby in ["age_group_rki", None]:
            depends_on = create_path_to_share_known_cases_of_scenario(
                scenario_name, groupby
            )
            produces = create_path_to_share_known_cases_plot(scenario_name, groupby)
            title = (
                f"Share Known Cases {'By Age Group' if groupby else ''} in {nice_name}"
            )
            title = None
            parametrization.append((depends_on, title, groupby, produces))

    return "depends_on, title, groupby, produces", parametrization


_SIGNATURE, _PARAMETRIZATION = _create_parametrization()


@pytask.mark.depends_on(_MODULE_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_plot_share_known_cases_per_scenario(depends_on, title, groupby, produces):
    share_known_cases = pd.read_pickle(depends_on[0])
    fig, ax = plot_share_known_cases(share_known_cases, title, groupby=groupby)
    fig.savefig(produces)
    plt.close()
