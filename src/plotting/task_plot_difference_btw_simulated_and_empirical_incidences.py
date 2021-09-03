import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.plotting.plotting import BLUE
from src.plotting.plotting import plot_incidences
from src.simulation.scenario_config import create_path_to_scenario_outcome_time_series

_DEPENDENCIES = {
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
    "plotting.py": SRC / "plotting" / "plotting.py",
    "empirical": BLD / "data" / "empirical_data_for_plotting.pkl",
}


SIM_DATA_PATH = create_path_to_scenario_outcome_time_series(
    scenario_name="combined_baseline", entry="new_known_case"
)

if SIM_DATA_PATH.exists():
    _DEPENDENCIES["simulated"] = SIM_DATA_PATH


@pytask.mark.skipif(
    not SIM_DATA_PATH.exists(), reason="combined_baseline data does not exist."
)
@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.produces(
    BLD / "figures" / "diff_btw_simulated_and_empirical_case_numbers.pdf"
)
def task_plot_difference_btw_official_and_simulated_cases(depends_on, produces):
    empirical = pd.read_pickle(depends_on["empirical"])
    simulated = pd.read_pickle(depends_on["simulated"])
    diff = simulated.subtract(empirical.loc[simulated.index, "new_known_case"], axis=0)
    fig, ax = plot_incidences(
        incidences={"": diff},
        title="Difference Between Simulated and Empirical Case Numbers",
        ylabel="difference between simulated and empirical case numbers",
        name_to_label={"": ""},
        colors=[BLUE],
        n_single_runs=-1,
    )
    ax.set_xlim(pd.Timestamp("2020-09-01"), pd.Timestamp("2021-07-01"))
    fig.tight_layout()
    fig.savefig(produces)
    plt.close()
