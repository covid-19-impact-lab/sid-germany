"""Basic scenarios for the October to Christmas period."""
import pandas as pd
import pytask
from sid import get_simulate_func

from src.config import BLD
from src.config import FAST_FLAG
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (  # noqa
    create_initial_conditions,
)
from src.policies.combine_policies_over_periods import get_october_to_christmas_policies
from src.simulation.main_specification import build_main_scenarios
from src.simulation.main_specification import FALL_PATH

NESTED_PARAMETRIZATION = build_main_scenarios(FALL_PATH)
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
}
if FAST_FLAG:
    DEPENDENCIES["initial_states"] = BLD / "data" / "debug_initial_states.parquet"


@pytask.mark.skip
@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.parametrize("produces, scenario, seed", PARAMETRIZATION)
def task_simulate_main_fall_scenario(depends_on, produces, scenario, seed):
    start_date = pd.Timestamp("2020-10-15")
    end_date = pd.Timestamp("2020-11-15") if FAST_FLAG else pd.Timestamp("2020-12-23")

    init_start = start_date - pd.Timedelta(31, unit="D")
    init_end = start_date - pd.Timedelta(1, unit="D")

    initial_states = pd.read_parquet(depends_on["initial_states"])
    share_known_cases = pd.read_pickle(depends_on["share_known_cases"])
    params = pd.read_pickle(depends_on["params"])

    initial_conditions = create_initial_conditions(
        start=init_start,
        end=init_end,
        seed=344490,
        reporting_delay=5,
    )
    contact_models = get_all_contact_models()
    policies = get_october_to_christmas_policies(
        contact_models=contact_models, **scenario
    )
    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=contact_models,
        contact_policies=policies,
        duration={"start": start_date, "end": end_date},
        initial_conditions=initial_conditions,
        share_known_cases=share_known_cases,
        path=produces.parent,
        seed=seed,
        saved_columns={
            "initial_states": ["age_group_rki"],
            "disease_states": ["newly_infected"],
            "time": ["date"],
            "other": ["new_known_case"],
        },
    )
    simulate(params)
