"""Basic prognosis 6 weeks into the future."""
import pandas as pd
import pytask
from sid import get_simulate_func

from src.config import BLD
from src.config import FAST_FLAG
from src.config import SRC
from src.create_initial_states.create_initial_conditions import (  # noqa
    create_initial_conditions,
)
from src.policies.combine_policies_over_periods import create_scenario_policies
from src.policies.combine_policies_over_periods import get_enacted_policies_of_2021
from src.policies.policy_tools import combine_dictionaries
from src.simulation.main_specification import build_main_scenarios
from src.simulation.main_specification import get_simulation_kwargs
from src.simulation.main_specification import PREDICT_PATH
from src.simulation.main_specification import SCENARIO_START


NESTED_PARAMETRIZATION = build_main_scenarios(PREDICT_PATH)
PARAMETRIZATION = [
    spec for seed_list in NESTED_PARAMETRIZATION.values() for spec in seed_list
]
"""Each specification consists of a produces path, the scenario dictioary and a seed"""

DEPENDENCIES = {
    "initial_states": BLD / "data" / "initial_states.parquet",
    "share_known_cases": BLD
    / "data"
    / "processed_time_series"
    / "share_known_cases.pkl",
    "params": SRC / "simulation" / "estimated_params.pkl",
    "contacts_py": SRC / "contact_models" / "get_contact_models.py",
    "policies_py": SRC / "policies" / "combine_policies_over_periods.py",
    "specs": SRC / "simulation" / "main_specification.py",
    "rki_data": BLD / "data" / "processed_time_series" / "rki.pkl",
    "synthetic_data_path": BLD / "data" / "initial_states.parquet",
    "test_shares_by_age_group": BLD
    / "data"
    / "testing"
    / "test_shares_by_age_group.pkl",
    "positivity_rate_by_age_group": BLD
    / "data"
    / "testing"
    / "positivity_rate_by_age_group.pkl",
    "positivity_rate_overall": BLD / "data" / "testing" / "positivity_rate_overall.pkl",
}
if FAST_FLAG:
    DEPENDENCIES["initial_states"] = BLD / "data" / "debug_initial_states.parquet"


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.parametrize("produces, scenario, seed", PARAMETRIZATION)
def task_simulate_main_prediction(depends_on, produces, scenario, seed):
    start_date = pd.Timestamp("2021-02-01")

    end_date = start_date + pd.Timedelta(weeks=4 if FAST_FLAG else 8)
    init_start = start_date - pd.Timedelta(31, unit="D")
    init_end = start_date - pd.Timedelta(1, unit="D")

    initial_conditions = create_initial_conditions(
        start=init_start,
        end=init_end,
        seed=3930,
        reporting_delay=5,
    )

    kwargs = get_simulation_kwargs(
        depends_on, init_start, end_date, extend_ars_dfs=True
    )

    scenario_name = produces.parent.name
    enacted_policies = get_enacted_policies_of_2021(
        contact_models=kwargs["contact_models"],
        scenario_start=SCENARIO_START,
    )
    scenario_policies = create_scenario_policies(
        contact_models=kwargs["contact_models"],
        prefix=scenario_name,
        start_date=SCENARIO_START,
        end_date=end_date,
        **scenario,
    )
    policies = combine_dictionaries([enacted_policies, scenario_policies])

    simulate = get_simulate_func(
        **kwargs,
        contact_policies=policies,
        duration={"start": start_date, "end": end_date},
        initial_conditions=initial_conditions,
        path=produces.parent,
        seed=seed,
        saved_columns={
            "initial_states": ["age_group_rki"],
            "disease_states": ["newly_infected"],
            "time": ["date"],
            "other": ["new_known_case"],
        },
    )
    simulate(kwargs["params"])
