import warnings
from functools import partial

import pandas as pd

from src.calculate_moments import calculate_period_outcome_sim
from src.config import BLD
from src.config import SID_DEPENDENCIES
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (
    create_initial_conditions,
)
from src.policies.policy_tools import combine_dictionaries
from src.simulation import scenario_simulation_inputs
from src.simulation.calculate_susceptibility import calculate_susceptibility
from src.simulation.seasonality import seasonality_model
from src.testing.shared import get_piecewise_linear_interpolation
from src.testing.testing_models import allocate_tests
from src.testing.testing_models import demand_test
from src.testing.testing_models import process_tests


def load_simulation_inputs(
    scenario,
    start_date,
    end_date,
    debug,
    group_share_known_case_path=None,
    period_outputs=False,
):
    """Load the simulation inputs.

    Does **not** include: params, path, seed.

    Args:
        scenario (str): string specifying the scenario. A function with the
            same name must exist in src.simulation.scenario_simulation_inputs.
        start_date (str): date on which the simulation starts. Data must be available
            for at least a month before the start date for the burn in period.
        end_date (str): date on which the simulation ends.
        debug (bool): Whether to use the debug or the full initial states.
        group_share_known_case_path (pathlib.Path, str or None): if not None, the group
            share known cases are loaded from this path and used for the creation of the
            initial conditions.
        period_outputs (bool, optional): whether to use period_outputs instead of saving
            the time series. Default is False.

    Returns:
        dict: Dictionary with most arguments of get_simulate_func. Keys are:
            - initial_states
            - contact_models
            - duration
            - events
            - saved_columns
            - virus_strains
            - derived_state_variables
            - seasonality_factor_model
            - initial_conditions
            - susceptibility_factor_model
            - testing_demand_models
            - testing_allocation_models
            - testing_processing_models
            - period_outputs
            - return_last_states
            - return_time_series

            - contact_policies
            - vaccination_models
            - rapid_test_models
            - rapid_test_reaction_models

    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    paths = get_simulation_dependencies(debug=debug)

    initial_states = pd.read_parquet(paths["initial_states"])
    contact_models = get_all_contact_models()

    # process dates
    one_day = pd.Timedelta(1, unit="D")
    init_start = start_date - pd.Timedelta(31, unit="D")
    init_end = start_date - one_day
    duration = {"start": start_date, "end": end_date}

    # testing models
    share_of_tests_for_symptomatics_series = pd.read_pickle(
        paths["share_of_tests_for_symptomatics_series"]
    )
    test_start = init_start - one_day
    test_end = end_date + one_day
    test_demand_func = partial(
        demand_test,
        share_of_tests_for_symptomatics_series=share_of_tests_for_symptomatics_series,
    )
    testing_demand_models = {
        "symptoms": {
            "model": test_demand_func,
            "start": test_start,
            "end": test_end,
        }
    }
    testing_allocation_models = {
        "direct_allocation": {
            "model": allocate_tests,
            "start": test_start,
            "end": test_end,
        }
    }
    testing_processing_models = {
        "direct_processing": {
            "model": process_tests,
            "start": test_start,
            "end": test_end,
        }
    }

    saved_columns = {
        "initial_states": ["age_group_rki"],
        "disease_states": ["newly_infected", "newly_deceased", "ever_infected"],
        "time": ["date"],
        "other": [
            "new_known_case",
            "virus_strain",
            "n_has_infected",
            "channel_infected_by_contact",
            "state",
            "knows_currently_infected",
            "currently_infected",
        ],
    }

    virus_shares = pd.read_pickle(paths["virus_shares"])
    rki_infections = pd.read_pickle(paths["rki"])

    group_weights = pd.read_pickle(paths["rki_age_groups"])["weight"]
    if group_share_known_case_path is not None:
        group_share_known_cases = pd.read_pickle(group_share_known_case_path)
    else:
        group_share_known_cases = None

    params = pd.read_pickle(paths["params"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        params_slice = params.loc[("share_known_cases", "share_known_cases")]
    overall_share_known_cases = get_piecewise_linear_interpolation(params_slice)

    initial_conditions = create_initial_conditions(
        start=init_start,
        end=init_end,
        seed=3930,
        reporting_delay=5,
        synthetic_data=initial_states[["county", "age_group_rki"]],
        empirical_infections=rki_infections,
        virus_shares=virus_shares,
        overall_share_known_cases=overall_share_known_cases,
        group_share_known_cases=group_share_known_cases,
        group_weights=group_weights,
    )

    seasonality_factor_model = partial(seasonality_model, contact_models=contact_models)

    def _currently_infected(df):
        return df["infectious"] | df["symptomatic"] | (df["cd_infectious_true"] >= 0)

    def _knows_currently_infected(df):
        return df["knows_immune"] & df["currently_infected"]

    derived_state_variables = {
        "currently_infected": _currently_infected,
        "knows_currently_infected": _knows_currently_infected,
    }

    fixed_inputs = {
        "initial_states": initial_states,
        "contact_models": contact_models,
        "duration": duration,
        "events": None,
        "testing_demand_models": testing_demand_models,
        "testing_allocation_models": testing_allocation_models,
        "testing_processing_models": testing_processing_models,
        "saved_columns": saved_columns,
        "initial_conditions": initial_conditions,
        "susceptibility_factor_model": calculate_susceptibility,
        "virus_strains": ["base_strain", "b117"],
        "seasonality_factor_model": seasonality_factor_model,
        "derived_state_variables": derived_state_variables,
    }

    if period_outputs:
        fixed_inputs["period_outputs"] = create_period_outputs()
        fixed_inputs["return_last_states"] = False
        fixed_inputs["return_time_series"] = False

    scenario_func = getattr(scenario_simulation_inputs, scenario)
    scenario_inputs = scenario_func(paths, fixed_inputs)
    simulation_inputs = combine_dictionaries([fixed_inputs, scenario_inputs])
    return simulation_inputs


def get_simulation_dependencies(debug):
    """Collect paths on which the simulation depends.

    This contains both paths to python modules and data paths.
    It only covers sid-germany specific paths, i.e. not sid.

    Args:
        debug (bool): Whether to use the debug initial states.

    Returns:
        paths (dict): Dictionary with the dependencies for the simulation.

    """
    if debug:
        initial_states_path = BLD / "data" / "debug_initial_states.parquet"
    else:
        initial_states_path = BLD / "data" / "initial_states.parquet"

    out = {
        **SID_DEPENDENCIES,
        "initial_states": initial_states_path,
        # to ensure that the checks on the initial states run before the
        # simulations we add the output of task_check_initial_states here
        # even though we don't use it.
        "output_of_check_initial_states": BLD
        / "data"
        / "comparison_of_age_group_distributions.png",
        "contact_models.py": SRC / "contact_models" / "get_contact_models.py",
        "contact_policies.py": SRC / "policies" / "enacted_policies.py",
        "testing_models.py": SRC / "testing" / "testing_models.py",
        "share_of_tests_for_symptomatics_series": BLD
        / "data"
        / "testing"
        / "share_of_tests_for_symptomatics_series.pkl",
        "initial_conditions.py": SRC
        / "create_initial_states"
        / "create_initial_conditions.py",
        "initial_infections.py": SRC
        / "create_initial_states"
        / "create_initial_infections.py",
        "initial_immunity.py": SRC
        / "create_initial_states"
        / "create_initial_immunity.py",
        "susceptibility_factor_model.py": SRC
        / "simulation"
        / "calculate_susceptibility.py",
        "virus_shares": BLD / "data" / "virus_strains" / "virus_shares_dict.pkl",
        "find_people_to_vaccinate.py": SRC / "policies" / "find_people_to_vaccinate.py",
        "vaccination_shares": BLD
        / "data"
        / "vaccinations"
        / "vaccination_shares_extended.pkl",
        "rapid_tests.py": SRC / "testing" / "rapid_tests.py",
        "rapid_test_reactions.py": SRC / "testing" / "rapid_test_reactions.py",
        "seasonality.py": SRC / "simulation" / "seasonality.py",
        "params": BLD / "params.pkl",
        "scenario_simulation_inputs.py": SRC
        / "simulation"
        / "scenario_simulation_inputs.py",
        "params_scenarios.py": SRC / "simulation" / "params_scenarios.py",
        "rki": BLD / "data" / "processed_time_series" / "rki.pkl",
        "rki_age_groups": BLD / "data" / "population_structure" / "age_groups_rki.pkl",
        "load_simulation_inputs.py": SRC / "simulation" / "load_simulation_inputs.py",
        "load_params.py": SRC / "simulation" / "load_params.py",
        "calculate_moments.py": SRC / "calculate_moments.py",
        # not strictly necessary because changes to scenario_config would change the
        # parametrization but for safety put it here
        "scenario_config.py": SRC / "simulation" / "scenario_config.py",
        "testing_shared.py": SRC / "testing" / "shared.py",
        "policy_tools.py": SRC / "policies" / "policy_tools.py",
    }

    return out


def create_period_outputs():
    outcomes = [
        "newly_infected",
        "new_known_case",
        "currently_infected",
        "knows_currently_infected",
        "ever_vaccinated",
        "newly_deceased",
    ]
    groupbys = ["state", "age_group_rki", None]

    period_outputs = {}

    for outcome in outcomes:
        for groupby in groupbys:
            gb_str = f"_by_{groupby}" if groupby is not None else ""
            period_outputs[f"{outcome}{gb_str}"] = partial(
                calculate_period_outcome_sim,
                outcome=outcome,
                groupby=groupby,
            )

    return period_outputs
