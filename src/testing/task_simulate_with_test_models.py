"""Fast simulation with positive test results distributed randomly among infected."""
from functools import partial

import pandas as pd
import pytask
from sid import get_simulate_func
from sid import load_epidemiological_parameters

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
    "initial_states": BLD / "data" / "initial_states.parquet",
    "params": SRC / "simulation" / "estimated_params.pkl",
    "positivity_rate_overall": BLD / "data" / "testing" / "share_tests_positive.csv",
    "testing_models": SRC / "testing" / "testing_models.py",
    "rki_data": BLD / "data" / "processed_time_series" / "rki.pkl",
    "synthetic_data_path": BLD / "data" / "initial_states.parquet",
}

OUT_PATH = BLD / "simulations" / "develop_testing_model"

PARAMETRIZATION = []
SEEDS = [200_500, 400_500, 600_500, 800_500, 1_000_500, 1_200_500][:1]
for i, seed in enumerate(SEEDS):
    PARAMETRIZATION += [
        (None, seed, OUT_PATH / f"with_models_stay_home_{i}" / "time_series"),
        (1.0, seed, OUT_PATH / f"with_models_meet_when_positive_{i}" / "time_series"),
    ]


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.parametrize("multiplier, seed, produces", PARAMETRIZATION)
def task_simulate_with_test_models(depends_on, multiplier, seed, produces):
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
    population_proportions = initial_states["age_group_rki"].value_counts(
        normalize=True
    )
    positivity_rate_by_age_group = pd.DataFrame(index=positivity_rate_overall.index)
    for g in groups:
        positivity_rate_by_age_group[g] = positivity_rate_overall

    params = pd.read_pickle(depends_on["params"])
    params = _adjust_params_to_testing_and_scenario(params, multiplier)
    # update params to use the new epi params
    current_epi_params = load_epidemiological_parameters()
    epi_param_names = current_epi_params.index.get_level_values("category")
    # drop old epi params
    params = params[~params.index.get_level_values("category").isin(epi_param_names)]
    # add new epi params
    params = pd.concat([params, current_epi_params])

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

    demand_test_func = partial(
        demand_test,
        share_known_cases=share_known_cases,
        positivity_rate_overall=positivity_rate_overall,
        test_shares_by_age_group=population_proportions,
        positivity_rate_by_age_group=positivity_rate_by_age_group,
        share_symptomatic_requesting_test=0.0,  # to have uniform testing
    )
    testing_demand_models = {"symptoms": {"model": demand_test_func}}
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
        seed=seed,
        saved_columns={
            "initial_states": ["age_group_rki", "cd_received_test_result_true_draws"],
            "disease_states": ["newly_infected", "symptomatic"],
            "time": ["date"],
            "other": [
                "allocated_test",
                "cd_received_test_result_true",
                "cd_infectious_true",
                "demands_test",
                "knows_immune",
                "new_known_case",
                "pending_test",
                "pending_test_date",
                "received_test_result",
                "to_be_processed_test",
            ],
        },
    )
    sim_func(params)


def _adjust_params_to_testing_and_scenario(params, multiplier):
    params = params.copy(deep=True)
    params.loc[("testing", "allocation", "rel_available_tests"), "value"] = 100_000
    params.loc[("testing", "processing", "rel_available_capacity"), "value"] = 100_000
    if multiplier is not None:
        subcategories = params.index.get_level_values("subcategory")
        test_reaction_params = subcategories == "positive_test_multiplier"
        params.loc[test_reaction_params] = 1.0
    return params
