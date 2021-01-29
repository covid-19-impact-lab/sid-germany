"""Basic prognosis 6 weeks into the future."""
from datetime import datetime

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
from src.policies.combine_policies_over_periods import get_jan_to_april_2021_policies
from src.simulation.base_prognosis.base_prognosis_specification import (
    build_base_prognosis_parametrization,
)


START_DATE = (pd.Timestamp(datetime.today()) - pd.Timedelta(days=14)).normalize()
END_DATE = START_DATE + pd.Timedelta(weeks=4 if FAST_FLAG else 8)
NESTED_PARAMETRIZATION = build_base_prognosis_parametrization()
SIMULATION_PARAMETRIZATION = [
    (other_multiplier, seed, produces)
    for other_multiplier, val in NESTED_PARAMETRIZATION.items()
    for seed, produces in val
]


DEPENDENCIES = {
    "initial_states": BLD / "data" / "initial_states.parquet",
    "share_known_cases": BLD
    / "data"
    / "processed_time_series"
    / "share_known_cases.pkl",
    "params": SRC / "simulation" / "estimated_params.pkl",
    "contacts_py": SRC / "contact_models" / "get_contact_models.py",
    "policies_py": SRC / "policies" / "combine_policies_over_periods.py",
}
if FAST_FLAG:
    DEPENDENCIES["initial_states"] = BLD / "data" / "debug_initial_states.parquet"


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.parametrize("other_multiplier, seed, produces", SIMULATION_PARAMETRIZATION)
def task_run_base_prognoses(depends_on, other_multiplier, seed, produces):
    init_start = START_DATE - pd.Timedelta(31, unit="D")
    init_end = START_DATE - pd.Timedelta(1, unit="D")

    initial_states = pd.read_parquet(depends_on["initial_states"])
    share_known_cases = pd.read_pickle(depends_on["share_known_cases"])
    params = pd.read_pickle(depends_on["params"])

    initial_conditions = create_initial_conditions(
        start=init_start,
        end=init_end,
        seed=3930,
        reporting_delay=5,
    )

    contact_models = get_all_contact_models(
        christmas_mode=None, n_extra_contacts_before_christmas=None
    )
    policies = get_jan_to_april_2021_policies(
        contact_models=contact_models,
        other_multiplier=other_multiplier,
        start_date=START_DATE,
        end_date=END_DATE,
        work_multiplier=0.7,
    )
    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=contact_models,
        contact_policies=policies,
        duration={"start": START_DATE, "end": END_DATE},
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
