"""Fast simulation with positive test results distributed randomly among infected."""
import pandas as pd
import pytask
from sid import get_simulate_func

from src.config import BLD
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (  # noqa
    create_initial_conditions,
)
from src.policies.combine_policies_over_periods import get_october_to_christmas_policies

DEPENDENCIES = {
    "share_known_cases": BLD
    / "data"
    / "processed_time_series"
    / "share_known_cases.pkl",
    "initial_states": BLD / "data" / "debug_initial_states.parquet",
    "params": SRC / "simulation" / "estimated_params.pkl",
}

OUT_PATH = BLD / "simulations" / "develop_testing_model"
PARAMETRIZATION = [
    (None, OUT_PATH / "without_models_stay_home" / "time_series"),
    (1.0, OUT_PATH / "without_models_meet_when_positive" / "time_series"),
]


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.parametrize("multiplier, produces", PARAMETRIZATION)
def task_simulate_with_raining_tests(depends_on, multiplier, produces):
    share_known_cases = pd.read_pickle(depends_on["share_known_cases"])
    initial_states = pd.read_pickle(depends_on["initial_states"])
    params = pd.read_pickle(depends_on["params"])
    if multiplier is not None:
        subcategories = params.index.get_level_values("subcategory")
        test_reaction_params = subcategories == "positive_test_multiplier"
        params.loc[test_reaction_params] = 1.0

    start_date = pd.Timestamp("2020-10-15")
    end_date = pd.Timestamp("2020-11-15")

    init_start = start_date - pd.Timedelta(31, unit="D")
    init_end = start_date - pd.Timedelta(1, unit="D")

    initial_conditions = create_initial_conditions(
        start=init_start,
        end=init_end,
        seed=22994,
        reporting_delay=5,
    )
    contact_models = get_all_contact_models()
    policies = get_october_to_christmas_policies(contact_models=contact_models)
    sim_func = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=contact_models,
        contact_policies=policies,
        duration={"start": start_date, "end": end_date},
        initial_conditions=initial_conditions,
        share_known_cases=share_known_cases,
        path=produces.parent,
        seed=898,
        saved_columns={
            "initial_states": ["age_group_rki"],
            "disease_states": ["newly_infected"],
            "time": ["date"],
            "other": ["new_known_case"],
        },
    )
    sim_func(params)
