import pandas as pd
import pytask

from src.config import SRC
from src.simulation.scenario_config import (
    create_path_to_initial_group_share_known_cases,
)
from src.simulation.scenario_config import create_path_to_share_known_cases_of_scenario

_DEPENDENCIES = {
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
    "snk": create_path_to_share_known_cases_of_scenario("fall_baseline"),
}


@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.produces(create_path_to_initial_group_share_known_cases("fall_baseline"))
def task_create_initial_group_share_known_cases(depends_on, produces):
    share_known_cases = pd.read_pickle(depends_on["snk"])
    share_known_cases = share_known_cases["mean"]
    share_known_cases = share_known_cases.unstack()

    len_interval = pd.Timedelta(days=28)
    first_date = share_known_cases.index.min()
    last_date = share_known_cases.index.max()
    start_date = max(last_date - len_interval, first_date)
    dates = pd.date_range(start_date, last_date)
    avg_group_share_known_cases = share_known_cases.loc[dates].mean(axis=0)
    avg_group_share_known_cases.to_pickle(produces)
