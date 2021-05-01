import matplotlib.pyplot as plt
import pandas as pd
import pytask
import sid
from packaging import version

from src.config import BLD
from src.config import FAST_FLAG
from src.policies.combine_policies_over_periods import get_october_to_christmas_policies
from src.simulation.main_specification import build_main_scenarios
from src.simulation.main_specification import FALL_PATH
from src.simulation.main_specification import load_simulation_inputs
from src.simulation.main_specification import SIMULATION_DEPENDENCIES


NESTED_PARAMETRIZATION = build_main_scenarios(FALL_PATH)
SCENARIOS = {
    name: NESTED_PARAMETRIZATION[name][0][1] for name in NESTED_PARAMETRIZATION
}
PARAMETRIZATION = [
    (FALL_PATH.joinpath(f"gantt_chart_{name}.png"), scenario)
    for name, scenario in SCENARIOS.items()
]
"""Each specification consists of a produces path, the scenario dictionary and a seed"""


if FAST_FLAG == "debug":
    SIMULATION_DEPENDENCIES["initial_states"] = (
        BLD / "data" / "debug_initial_states.parquet"
    )


@pytask.mark.skipif(
    version.parse(sid.__version__) < version.parse("0.0.5"),
    reason="gantt chart is later implemented.",
)
@pytask.mark.depends_on(SIMULATION_DEPENDENCIES)
@pytask.mark.parametrize(
    "produces, scenario",
    PARAMETRIZATION,
)
def task_plot_policy_gantt_chart_of_main_fall_scenario(depends_on, produces, scenario):
    # determine dates
    start_date = pd.Timestamp("2020-10-15")

    early_end_date = pd.Timestamp("2020-11-15")
    late_end_date = pd.Timestamp("2020-12-23")
    if FAST_FLAG == "debug":
        end_date = early_end_date
    else:
        end_date = late_end_date

    init_start = start_date - pd.Timedelta(31, unit="D")

    _, simulation_inputs = load_simulation_inputs(
        depends_on,
        init_start,
        end_date,
    )

    policies = get_october_to_christmas_policies(
        contact_models=simulation_inputs["contact_models"], **scenario
    )

    ax = sid.plotting.plot_policy_gantt_chart(policies)  # noqa: F841

    plt.savefig(produces)
