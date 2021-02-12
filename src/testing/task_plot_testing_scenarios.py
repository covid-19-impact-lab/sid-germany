import dask.dataframe as dd
import pytask

from src.config import BLD
from src.config import SRC
from src.testing.plotting_for_test_models import plot_known_vs_total_cases
from src.testing.plotting_for_test_models import plot_share_known_cases
from src.testing.task_simulate_with_test_models import SEEDS

OUT_PATH = BLD / "simulations" / "develop_testing_model"

SCENARIOS = [
    "with_models_stay_home",
    "with_models_meet_when_positive",
]

DEPENDENCIES = {("plotting_module", 0): SRC / "testing" / "plotting_for_test_models.py"}
for scenario in SCENARIOS:
    for i in range(len(SEEDS)):
        DEPENDENCIES[(scenario, i)] = OUT_PATH / f"{scenario}_{i}" / "time_series"

FIGURE_PATHS = {}
for scenario in SCENARIOS:
    FIGURE_PATHS[f"{scenario}_known_vs_total"] = (
        OUT_PATH / f"{scenario}_known_vs_total.png"
    )
    FIGURE_PATHS[f"{scenario}_share_known_cases"] = (
        OUT_PATH / f"{scenario}_share_known_cases.png"
    )


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.produces(FIGURE_PATHS)
def task_plot_scenarios(depends_on, produces):
    for scenario in SCENARIOS:
        dfs = [dd.read_parquet(depends_on[(scenario, i)]) for i in range(len(SEEDS))]
        title = scenario.replace("_", " ").title()
        fig, ax = plot_known_vs_total_cases(dfs, title)
        fig.savefig(produces[f"{scenario}_known_vs_total"])
        fig, ax = plot_share_known_cases(dfs, title)
        fig.savefig(produces[f"{scenario}_share_known_cases"])
