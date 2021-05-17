import pandas as pd
import pytask

from src.config import BLD
from src.config import FAST_FLAG
from src.simulation.task_process_simulation_outputs import (
    create_path_for_share_known_cases_of_scenario,
)
from src.simulation.task_run_simulation import NAMED_SCENARIOS


def create_initial_group_share_known_cases_path(name, fast_flag, date):
    file_name = f"{fast_flag}_{name}_for_{date.date()}.pkl"
    return BLD / "simulations" / "share_known_case_prediction" / file_name


def create_parametrization(named_scenarios, fast_flag):
    parametrization = []
    for name, spec in named_scenarios.items():
        if "baseline" in name:
            depends_on = create_path_for_share_known_cases_of_scenario(name, fast_flag)
            date = pd.Timestamp(spec["end_date"])
            produces = create_initial_group_share_known_cases_path(
                name, fast_flag, date
            )
            parametrization.append((depends_on, date, produces))
    return "depends_on, date, produces", parametrization


@pytask.mark.parametrize(*create_parametrization(NAMED_SCENARIOS, FAST_FLAG))
def task_create_initial_group_share_known_cases(depends_on, date, produces):
    share_known_cases = pd.read_pickle(depends_on)
    share_known_cases = share_known_cases["mean"]
    share_known_cases = share_known_cases.unstack()

    len_interval = pd.Timedelta(days=28)
    dates = pd.date_range(date - len_interval, date)
    avg_group_share_known_cases = share_known_cases.loc[dates].mean(axis=0)
    avg_group_share_known_cases.to_pickle(produces)
