"""Build the specifiation for the base prognosis."""
from functools import partial

import pandas as pd

from src.config import BLD
from src.config import FAST_FLAG
from src.config import SHARE_REFUSE_VACCINATION
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.policies.find_people_to_vaccinate import find_people_to_vaccinate
from src.simulation.calculate_susceptibility import calculate_susceptibility
from src.simulation.seasonality import seasonality_model
from src.testing.rapid_tests import rapid_test_demand
from src.testing.rapid_tests import rapid_test_reactions
from src.testing.testing_models import allocate_tests
from src.testing.testing_models import demand_test
from src.testing.testing_models import process_tests

FALL_PATH = BLD / "simulations" / f"{FAST_FLAG}_main_fall_scenarios"
PREDICT_PATH = BLD / "simulations" / f"{FAST_FLAG}_main_predictions"

SCENARIO_START = pd.Timestamp("2021-04-30")


SIMULATION_DEPENDENCIES = {
    "initial_states": BLD / "data" / "initial_states.parquet",
    "vaccination_shares": BLD
    / "data"
    / "vaccinations"
    / "vaccination_shares_extended.pkl",
    "params": BLD / "params.pkl",
    "rki_data": BLD / "data" / "processed_time_series" / "rki.pkl",
    "synthetic_data_path": BLD / "data" / "initial_states.parquet",
    "characteristics_of_the_tested": BLD
    / "data"
    / "testing"
    / "characteristics_of_the_tested.csv",
    "virus_shares": BLD / "data" / "virus_strains" / "final_strain_shares.pkl",
    # py files
    "contacts_py": SRC / "contact_models" / "get_contact_models.py",
    "testing_py": SRC / "testing" / "testing_models.py",
    "rapid_tests_py": SRC / "testing" / "rapid_tests.py",
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
    "seasonality_py": SRC / "simulation" / "seasonality.py",
    "enacted_policies.py": SRC / "policies" / "enacted_policies.py",
}


def build_main_scenarios(base_path):
    """Build the nested scenario specifications.

    Args:
        base_path (pathlib.Path): Path where each simulation run will get
            a separate folder. (spring or fall)

    Returns:
        nested_parametrization (dict): Keys are the names of the scenarios.
            Values are lists of tuples. For each seed there is one tuple.
            Each tuple consists of:

            1. the path where sid will save the time series data.
            2. the scenario specification consisting of the educ and
               other multiplier and work_fill_value.
            3. the seed to be used by sid.

    """
    base_scenario = {}

    scenarios = {
        "base_scenario": base_scenario,
    }

    nested_parametrization = {}
    for name, scenario in scenarios.items():
        nested_parametrization[name] = []

        # rapid tests for leisure, school and most work settings
        # were very rare still in fall so we abstract from them completely.
        if "fall" in base_path.name:
            rapid_test_models = None
            rapid_test_reaction_models = None
        else:
            rapid_test_models = {
                "standard_rapid_test_demand": {
                    "model": rapid_test_demand,
                    "start": "2021-01-01",
                    "end": "2025-01-01",
                }
            }

            rapid_test_reaction_models = {
                "rapid_test_reactions": {
                    "model": rapid_test_reactions,
                    "start": "2021-01-01",
                    "end": "2025-01-01",
                }
            }

        if FAST_FLAG == "debug":
            n_seeds = 1
        elif FAST_FLAG == "full":
            n_seeds = 20
        elif FAST_FLAG == "verify":
            if "base" in name:
                n_seeds = 18
            else:
                n_seeds = 5
        else:
            raise ValueError(
                f"Unknown FAST_FLAG: {FAST_FLAG}. Must be one of "
                "'debug', 'verify', 'full'."
            )

        for i in range(n_seeds):
            seed = 300_000 + 700_001 * i
            produces = base_path / f"{name}_{seed}" / "time_series"
            nested_parametrization[name].append(
                (
                    produces,
                    scenario,
                    rapid_test_models,
                    rapid_test_reaction_models,
                    seed,
                )
            )

    return nested_parametrization


def load_simulation_inputs(depends_on, init_start, end_date):
    characteristics_of_the_tested = pd.read_csv(
        depends_on["characteristics_of_the_tested"],
        index_col="date",
        parse_dates=["date"],
    )

    share_of_tests_for_symptomatics_series = characteristics_of_the_tested[
        [
            "share_symptomatic_lower_bound_extrapolated",
            "share_symptomatic_among_known_extrapolated",
        ]
    ].mean(axis=1)

    simulation_inputs = _get_testing_models(
        init_start=init_start,
        end_date=end_date,
        share_of_tests_for_symptomatics_series=share_of_tests_for_symptomatics_series,
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

    def _currently_infected(df):
        return df["infectious"] | df["symptomatic"] | (df["cd_infectious_true"] >= 0)

    def _knows_currently_infected(df):
        return df["knows_immune"] & df["currently_infected"]

    simulation_inputs["params"] = params
    simulation_inputs["derived_state_variables"] = {
        "currently_infected": _currently_infected,
        "knows_currently_infected": _knows_currently_infected,
    }
    simulation_inputs["seasonality_factor_model"] = partial(
        seasonality_model, contact_models=simulation_inputs["contact_models"]
    )
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
    share_of_tests_for_symptomatics_series,
):
    demand_test_func = partial(
        demand_test,
        share_of_tests_for_symptomatics_series=share_of_tests_for_symptomatics_series,
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
