"""Build the specifiation for the base prognosis."""
from functools import partial

import pandas as pd

from src.config import BLD
from src.config import FAST_FLAG
from src.config import SHARE_REFUSE_VACCINATION
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.policies.combine_policies_over_periods import (
    get_educ_options_mid_march_to_easter,
)
from src.policies.combine_policies_over_periods import (
    strict_emergency_care,
)
from src.policies.find_people_to_vaccinate import find_people_to_vaccinate
from src.policies.policy_tools import combine_dictionaries
from src.simulation.calculate_susceptibility import calculate_susceptibility
from src.testing.testing_models import allocate_tests
from src.testing.testing_models import demand_test
from src.testing.testing_models import process_tests

FALL_PATH = BLD / "simulations" / f"{FAST_FLAG}_main_fall_scenarios"
PREDICT_PATH = BLD / "simulations" / f"{FAST_FLAG}_main_predictions"

SCENARIO_START = pd.Timestamp("2021-04-06")  # after Easter holidays


SIMULATION_DEPENDENCIES = {
    "initial_states": BLD / "data" / "initial_states.parquet",
    "vaccination_shares": BLD
    / "data"
    / "vaccinations"
    / "vaccination_shares_quadratic.pkl",
    "params": BLD / "params.pkl",
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
    "virus_shares": BLD / "data" / "virus_strains" / "final_strain_shares.pkl",
    # py files
    "contacts_py": SRC / "contact_models" / "get_contact_models.py",
    "policies_py": SRC / "policies" / "combine_policies_over_periods.py",
    "testing_py": SRC / "testing" / "testing_models.py",
    "specs_py": SRC / "simulation" / "main_specification.py",
    "initial_conditions_py": SRC
    / "create_initial_states"
    / "create_initial_conditions.py",
    "initial_infections_py": SRC
    / "create_initial_states"
    / "create_initial_infections.py",
    "initial_immunity_py": SRC / "create_initial_states" / "create_initial_immunity.py",
    # to ensure that the checks on the initial states run before the simulations
    # we add the output of task_check_initial_states here even though we don't use it.
    "output_of_check_initial_states": BLD
    / "data"
    / "comparison_of_age_group_distributions.png",
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
    if FAST_FLAG == "debug":
        n_seeds = 1
    elif FAST_FLAG == "full":
        n_seeds = 20
    elif FAST_FLAG == "verify":
        # with the verify fast flag only base scenario is run for the fall
        # -> 10 * 2 + 3 * 3 = 29
        if "base" in base_path.name:
            n_seeds = 10
        else:
            n_seeds = 3
    else:
        raise ValueError(
            f"Unknown FAST_FLAG: {FAST_FLAG}. Must be one of 'debug', 'verify', 'full'."
        )

    if "predictions" in base_path.name:
        base_scenario = combine_dictionaries(
            [{"educ_multiplier": 0.5}, get_educ_options_mid_march_to_easter()]
        )
    elif "fall" in base_path.name:
        base_scenario = {"educ_multiplier": 0.8}
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
    emergency_child_care = combine_dictionaries(
        [{"educ_multiplier": None}, strict_emergency_care()]
    )

    if FAST_FLAG == "debug" or (FAST_FLAG == "verify" and "fall" in base_path.name):
        scenarios = {"base_scenario": base_scenario}
    else:
        scenarios = {
            "base_scenario": base_scenario,
            "november_home_office_level": nov_home_office,
            "spring_home_office_level": spring_home_office,
            "emergency_child_care": emergency_child_care,
        }

    nested_parametrization = {}
    for name, scenario in scenarios.items():
        nested_parametrization[name] = []
        for i in range(n_seeds):
            seed = 300_000 + 700_001 * i
            produces = base_path / f"{name}_{seed}" / "time_series"
            nested_parametrization[name].append((produces, scenario, seed))

    return nested_parametrization


def load_simulation_inputs(depends_on, init_start, end_date, extend_ars_dfs=False):
    test_inputs = {
        "test_shares_by_age_group": pd.read_pickle(
            depends_on["test_shares_by_age_group"]
        ),
        "positivity_rate_by_age_group": pd.read_pickle(
            depends_on["positivity_rate_by_age_group"]
        ),
        "positivity_rate_overall": pd.read_pickle(
            depends_on["positivity_rate_overall"]
        ),
    }

    if extend_ars_dfs:
        for name, df in test_inputs.items():
            test_inputs[name] = _extend_df_into_future(df, end_date=end_date)

    simulation_inputs = _get_testing_models(
        init_start=init_start,
        end_date=end_date,
        **test_inputs,
    )
    simulation_inputs["initial_states"] = pd.read_parquet(depends_on["initial_states"])
    simulation_inputs["contact_models"] = get_all_contact_models()
    simulation_inputs["susceptibility_factor_model"] = calculate_susceptibility

    # Virus Variant Specification --------------------------------------------

    simulation_inputs["virus_strains"] = ["base_strain", "b117"]
    strain_shares = pd.read_pickle(depends_on["virus_shares"])
    virus_shares = {
        "base_strain": 1 - strain_shares["b117"],
        "b117": strain_shares["b117"],
    }

    params = pd.read_pickle(depends_on["params"])
    params.loc[("virus_strain", "base_strain", "factor")] = 1.0
    # source: https://doi.org/10.1101/2020.12.24.20248822
    # "We estimate that this variant has a 43–90%
    # (range of 95% credible intervals 38–130%) higher
    # reproduction number than preexisting variants"
    # currently we take the midpoint of 66%
    params.loc[("virus_strain", "b117", "factor")] = 1.67

    # Vaccination Model ----------------------------------------------------

    if init_start > pd.Timestamp("2021-01-01"):
        vaccination_shares = pd.read_pickle(depends_on["vaccination_shares"])
        simulation_inputs["vaccination_models"] = {
            "standard": {
                "model": partial(
                    find_people_to_vaccinate,
                    vaccination_shares=vaccination_shares,
                    no_vaccination_share=SHARE_REFUSE_VACCINATION,
                    init_start=init_start,
                )
            }
        }

    simulation_inputs["params"] = params
    return virus_shares, simulation_inputs


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
    positivity_rate_overall,
    test_shares_by_age_group,
    positivity_rate_by_age_group,
):
    demand_test_func = partial(
        demand_test,
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
