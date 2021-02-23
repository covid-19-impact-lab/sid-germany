"""Basic scenarios for the October to Christmas period."""
from functools import partial

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
from src.simulation.main_predictions.task_simulate_main_predictions import (
    SHARE_SYMPTOMATIC_REQUESTING_TEST,
)
from src.simulation.main_predictions.task_simulate_main_predictions import (
    update_params_to_new_epi_params,
)
from src.simulation.main_specification import build_main_scenarios
from src.simulation.main_specification import FALL_PATH
from src.testing.testing_models import allocate_tests
from src.testing.testing_models import demand_test
from src.testing.testing_models import process_tests


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
def task_simulate_main_fall_scenario(depends_on, produces, scenario, seed):
    # load dependencies
    initial_states = pd.read_parquet(depends_on["initial_states"])
    share_known_cases = pd.read_pickle(depends_on["share_known_cases"])

    test_shares_by_age_group = pd.read_pickle(depends_on["test_shares_by_age_group"])
    positivity_rate_by_age_group = pd.read_pickle(
        depends_on["positivity_rate_by_age_group"],
    )
    positivity_rate_overall = pd.read_pickle(depends_on["positivity_rate_overall"])

    params = pd.read_pickle(depends_on["params"])
    params = update_params_to_new_epi_params(params)

    # determine dates
    start_date = pd.Timestamp("2020-10-15")
    end_date = pd.Timestamp("2020-11-15") if FAST_FLAG else pd.Timestamp("2020-12-23")
    init_start = start_date - pd.Timedelta(31, unit="D")
    init_end = start_date - pd.Timedelta(1, unit="D")

    # create inputs for the simulate func
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

    demand_test_func = partial(
        demand_test,
        share_known_cases=share_known_cases,
        positivity_rate_overall=positivity_rate_overall,
        test_shares_by_age_group=test_shares_by_age_group,
        positivity_rate_by_age_group=positivity_rate_by_age_group,
        share_symptomatic_requesting_test=SHARE_SYMPTOMATIC_REQUESTING_TEST,
    )
    one_day = pd.Timedelta(1, unit="D")
    testing_demand_models = {
        "symptoms": {
            "model": demand_test_func,
            "start": init_start - one_day,
            "end": end_date + one_day,
        }
    }
    testing_allocation_models = {
        "direct_allocation": {
            "model": allocate_tests,
            "start": init_start - one_day,
            "end": end_date + one_day,
        }
    }
    testing_processing_models = {
        "direct_processing": {
            "model": process_tests,
            "start": init_start - one_day,
            "end": end_date + one_day,
        }
    }

    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=contact_models,
        contact_policies=policies,
        duration={"start": start_date, "end": end_date},
        initial_conditions=initial_conditions,
        testing_demand_models=testing_demand_models,
        testing_allocation_models=testing_allocation_models,
        testing_processing_models=testing_processing_models,
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
