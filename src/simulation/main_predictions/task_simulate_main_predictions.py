"""Basic prognosis 6 weeks into the future."""
from functools import partial

import pandas as pd
import pytask
from sid import get_simulate_func
from sid import load_epidemiological_parameters

from src.config import BLD
from src.config import FAST_FLAG
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (  # noqa
    create_initial_conditions,
)
from src.policies.combine_policies_over_periods import get_jan_to_april_2021_policies
from src.simulation.main_specification import build_main_scenarios
from src.simulation.main_specification import PREDICT_PATH
from src.testing.testing_models import allocate_tests
from src.testing.testing_models import demand_test
from src.testing.testing_models import process_tests

# ----------------------------------------------------------------------------
# At the moment we assume uniform testing!
SHARE_SYMPTOMATIC_REQUESTING_TEST = 0.0
# ----------------------------------------------------------------------------


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
    initial_states = pd.read_parquet(depends_on["initial_states"])
    share_known_cases = pd.read_pickle(depends_on["share_known_cases"])

    test_shares_by_age_group = pd.read_pickle(depends_on["test_shares_by_age_group"])
    positivity_rate_by_age_group = pd.read_pickle(
        depends_on["positivity_rate_by_age_group"],
    )
    positivity_rate_overall = pd.read_pickle(depends_on["positivity_rate_overall"])

    params = pd.read_pickle(depends_on["params"])
    params = _update_params_to_new_epi_params(params)

    # determine dates
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=15)).normalize()
    end_date = start_date + pd.Timedelta(weeks=4 if FAST_FLAG else 8)
    init_start = start_date - pd.Timedelta(31, unit="D")
    init_end = start_date - pd.Timedelta(1, unit="D")

    # extend arguments into the future
    test_shares_by_age_group = _extend_df_into_future(
        test_shares_by_age_group, end_date
    )
    positivity_rate_by_age_group = _extend_df_into_future(
        positivity_rate_by_age_group, end_date
    )
    positivity_rate_overall = _extend_df_into_future(positivity_rate_overall, end_date)

    # create inputs for the simulate func
    initial_conditions = create_initial_conditions(
        start=init_start,
        end=init_end,
        seed=3930,
        reporting_delay=5,
    )
    contact_models = get_all_contact_models()
    policies = get_jan_to_april_2021_policies(
        contact_models=contact_models,
        start_date=start_date,
        end_date=end_date,
        **scenario,
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


def _update_params_to_new_epi_params(params):
    """Update params to use the new epidemiological parameters.

    This function is only temporary and will be removed once the parameters
    are updated and newly estimated.

    The function also adds non binding capacities for testing.

    """
    current_epi_params = load_epidemiological_parameters()
    epi_param_names = current_epi_params.index.get_level_values("category")
    without_epi = params[
        ~params.index.get_level_values("category").isin(epi_param_names)
    ]
    new_params = pd.concat([without_epi, current_epi_params])

    new_params.loc[("testing", "allocation", "rel_available_tests")] = 100_000
    new_params.loc[("testing", "processing", "rel_available_capacity")] = 100_000
    return new_params


def _extend_df_into_future(short, end_date):
    future_dates = pd.date_range(short.index.max(), end_date, closed="right")
    extended_index = short.index.append(future_dates)
    extended = short.reindex(extended_index).fillna(method="ffill")
    return extended
