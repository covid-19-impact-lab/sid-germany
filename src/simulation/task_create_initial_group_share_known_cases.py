import pandas as pd
import pytask

from src.simulation.scenario_config import (
    create_path_to_initial_group_share_known_cases,
)
from src.simulation.scenario_config import create_path_to_share_known_cases_of_scenario
from src.simulation.scenario_config import get_named_scenarios


def _create_parametrization():
    named_scenarios = get_named_scenarios()
    parametrization = []
    for name, spec in named_scenarios.items():
        if "baseline" in name:
            depends_on = create_path_to_share_known_cases_of_scenario(name)
            date = pd.Timestamp(spec["end_date"])
            produces = create_path_to_initial_group_share_known_cases(name, date)
            parametrization.append((depends_on, date, produces))
    return "depends_on, date, produces", parametrization


_SIGNATURE, _PARAMETRIZATION = _create_parametrization()


@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_create_initial_group_share_known_cases(depends_on, date, produces):
    share_known_cases = pd.read_pickle(depends_on)
    share_known_cases = share_known_cases["mean"]
    share_known_cases = share_known_cases.unstack()

    len_interval = pd.Timedelta(days=28)
    dates = pd.date_range(date - len_interval, date)
    avg_group_share_known_cases = share_known_cases.loc[dates].mean(axis=0)
    avg_group_share_known_cases.to_pickle(produces)
