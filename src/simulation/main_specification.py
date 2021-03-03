"""Build the specifiation for the base prognosis."""
from functools import partial

import pandas as pd

from src.config import BLD
from src.config import FAST_FLAG
from src.contact_models.get_contact_models import get_all_contact_models
from src.policies.policy_tools import combine_dictionaries
from src.testing.testing_models import allocate_tests
from src.testing.testing_models import demand_test
from src.testing.testing_models import process_tests


FALL_PATH = BLD / "simulations" / "main_fall_scenarios"
PREDICT_PATH = BLD / "simulations" / "main_predictions"
SCENARIO_START = pd.Timestamp("2021-03-01")

PRIMARY_AND_GRADUATION_CLASSES = {
    "subgroup_query": "occupation == 'school' & (age < 12 | age in [16, 17, 18])",
    "others_attend": False,
    "hygiene_multiplier": 0.5,
}


def build_main_scenarios(base_path):
    """Build the nested scenario specifications.

    Args:
        base_path (pathlib.Path): Path where each simulation run will get
            a separate folder.

    Returns:
        nested_parametrization (dict): Keys are the names of the scenarios.
            Values are lists of tuples. For each seed there is one tuple.
            Each tuple consists of:
                1. the path where sid will save the time series data.
                2. the scenario specification consisting of the educ and
                   other multiplier and work_fill_value.
                3. the seed to be used by sid.

    """
    n_seeds = 1 if FAST_FLAG else 10

    if "predictions" in base_path.name:
        base_scenario = {
            "a_b_educ_options": {"school": PRIMARY_AND_GRADUATION_CLASSES},
            "educ_multiplier": 0.5,
        }
    elif "fall" in base_path.name:
        base_scenario = {
            "a_b_educ_options": {},
            "educ_multiplier": 0.8,
        }
    else:
        raise ValueError(
            f"Unknown situation: {base_path.name}. "
            "Only fall and predictions supported at the moment."
        )

    # November average work multiplier: 0.83
    # 1st lockdown (24.3.-08.04.) average work multiplier: 0.56
    nov_home_office = combine_dictionaries([base_scenario, {"work_fill_value": 0.83}])
    spring_home_office = combine_dictionaries(
        [base_scenario, {"work_fill_value": 0.56}]
    )
    schools_stay_closed = {"a_b_educ_options": {}, "educ_multiplier": 0.0}

    if FAST_FLAG:
        scenarios = {
            "base_scenario": base_scenario,
            "november_home_office_level": nov_home_office,
            "spring_home_office_level": spring_home_office,
            "keep_schools_closed": schools_stay_closed,
        }
    if not FAST_FLAG:
        scenarios = {
            "base_scenario": base_scenario,
            "november_home_office_level": nov_home_office,
            "spring_home_office_level": spring_home_office,
            "keep_schools_closed": schools_stay_closed,
        }

    nested_parametrization = {}
    for name, scenario in scenarios.items():
        nested_parametrization[name] = []
        for i in range(n_seeds):
            seed = 300_000 + 700_001 * i
            produces = base_path / f"{name}_{seed}" / "time_series"
            nested_parametrization[name].append((produces, scenario, seed))

    return nested_parametrization


def get_simulation_kwargs(depends_on, init_start, end_date, extend_ars_dfs=False):
    test_kwargs = {
        "test_shares_by_age_group": pd.read_pickle(
            depends_on["test_shares_by_age_group"]
        ),
        "positivity_rate_by_age_group": pd.read_pickle(
            depends_on["positivity_rate_by_age_group"]
        ),
        "positivity_rate_overall": pd.read_pickle(
            depends_on["positivity_rate_overall"]
        ),
        "share_known_cases": pd.read_pickle(depends_on["share_known_cases"]),
    }

    if extend_ars_dfs:
        for name, df in test_kwargs.items():
            test_kwargs[name] = _extend_df_into_future(df, end_date=end_date)

    kwargs = _get_testing_models(
        init_start=init_start,
        end_date=end_date,
        **test_kwargs,
    )
    kwargs["initial_states"] = pd.read_parquet(depends_on["initial_states"])
    kwargs["params"] = pd.read_pickle(depends_on["params"])
    kwargs["contact_models"] = get_all_contact_models()

    return kwargs


def _extend_df_into_future(df, end_date):
    """Take the last values of a DataFrame and propagate it into the future.

    Args:
        df (pandas.DataFrame): the index must be a DatetimeIndex.
        end_date (pandas.Timestamp): date until which the short DataFrame is
            to be extended.

    Returns:
        extended (pandas.DataFrame): the index runs from the start date of
            short to end_date. If there were NaN in short these were filled
            with the next available non NaN value.

    """
    future_dates = pd.date_range(df.index.max(), end_date, closed="right")
    extended_index = df.index.append(future_dates)
    extended = df.reindex(extended_index).fillna(method="ffill")
    return extended


def _get_testing_models(
    init_start,
    end_date,
    share_known_cases,
    positivity_rate_overall,
    test_shares_by_age_group,
    positivity_rate_by_age_group,
):
    demand_test_func = partial(
        demand_test,
        share_known_cases=share_known_cases,
        positivity_rate_overall=positivity_rate_overall,
        test_shares_by_age_group=test_shares_by_age_group,
        positivity_rate_by_age_group=positivity_rate_by_age_group,
    )
    one_day = pd.Timedelta(1, unit="D")
    testing_models = {
        "testing_demand_models": {
            "symptoms": {
                "model": demand_test_func,
                "start": init_start - one_day,
                "end": end_date + one_day,
            },
        },
        "testing_allocation_models": {
            "direct_allocation": {
                "model": allocate_tests,
                "start": init_start - one_day,
                "end": end_date + one_day,
            },
        },
        "testing_processing_models": {
            "direct_processing": {
                "model": process_tests,
                "start": init_start - one_day,
                "end": end_date + one_day,
            },
        },
    }
    return testing_models
