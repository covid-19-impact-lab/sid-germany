"""Fast simulation with positive test results distributed randomly among infected."""
from functools import partial

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
from src.testing.testing_models import allocate_tests
from src.testing.testing_models import demand_test
from src.testing.testing_models import process_tests


DEPENDENCIES = {
    "share_known_cases": BLD
    / "data"
    / "processed_time_series"
    / "share_known_cases.pkl",
    "initial_states": BLD / "data" / "debug_initial_states.parquet",
    "params": SRC / "simulation" / "estimated_params.pkl",
    "positivity_rate_overall": BLD
    / "data"
    / "processed_time_series"
    / "share_tests_positive.csv",
    "testing_models": SRC / "testing" / "testing_models.py",
}

OUT_PATH = BLD / "simulations" / "develop_testing_model"

PARAMETRIZATION = [
    (None, OUT_PATH / "with_models_stay_home" / "time_series"),
    (1.0, OUT_PATH / "with_models_meet_when_positive" / "time_series"),
]


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.parametrize("multiplier, produces", PARAMETRIZATION)
def task_simulate_with_test_models(depends_on, multiplier, produces):
    initial_states = pd.read_parquet(depends_on["initial_states"])

    share_known_cases = pd.read_pickle(depends_on["share_known_cases"])
    positivity_rate_overall = pd.read_csv(
        depends_on["positivity_rate_overall"], parse_dates=["date"]
    )
    positivity_rate_overall = positivity_rate_overall.set_index("date")[
        "share_tests_positive"
    ]

    # These must be filled in once we have the data from the ARS!
    groups = initial_states["age_group_rki"].unique()
    test_shares_by_age_group = pd.Series(data=1 / 6, index=groups)
    positivity_rate_by_age_group = pd.DataFrame(index=positivity_rate_overall.index)
    for g in groups:
        positivity_rate_by_age_group[g] = positivity_rate_overall

    params = pd.read_pickle(depends_on["params"])
    params = _adjust_params_to_testing_and_scenario(params, multiplier)

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

    testing_demand_models = {
        "symptoms": {
            "model": partial(
                demand_test,
                share_known_cases=share_known_cases,
                positivity_rate_overall=positivity_rate_overall,
                test_shares_by_age_group=test_shares_by_age_group,
                positivity_rate_by_age_group=positivity_rate_by_age_group,
            ),
        }
    }

    testing_allocation_models = {"direct_allocation": {"model": allocate_tests}}

    testing_processing_models = {"direct_processing": {"model": process_tests}}

    sim_func = get_simulate_func(
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
        seed=898,
        saved_columns={
            "initial_states": ["age_group_rki"],
            "disease_states": ["newly_infected"],
            "time": ["date"],
            "other": ["new_known_case"],
        },
    )
    sim_func(params)


def _adjust_params_to_testing_and_scenario(params, multiplier):
    params.loc[("testing", "allocation", "rel_available_tests"), "value"] = 100_000
    params.loc[("testing", "processing", "rel_available_capacity"), "value"] = 100_000
    if multiplier is not None:
        subcategories = params.index.get_level_values("subcategory")
        test_reaction_params = subcategories == "positive_test_multiplier"
        params.loc[test_reaction_params] = 1.0
    return params