from functools import partial
from pathlib import Path

import pandas as pd

from src.config import BLD
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (
    create_initial_conditions,
)
from src.policies.policy_tools import combine_dictionaries
from src.simulation import scenario_simulation_inputs
from src.simulation.calculate_susceptibility import calculate_susceptibility
from src.simulation.seasonality import seasonality_model
from src.testing.testing_models import allocate_tests
from src.testing.testing_models import demand_test
from src.testing.testing_models import process_tests


def load_simulation_inputs(scenario, start_date, end_date, debug):
    """Load the simulation inputs.

    Does **not** include: params, path, seed.

    Args:
        scenario (str): string specifying the scenario. A function with the
            same name must exist in src.simulation.scenario_simulation_inputs.

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
        ],
    }

    virus_shares = pd.read_pickle(paths["virus_shares"])

    initial_conditions = create_initial_conditions(
        start=init_start,
        end=init_end,
        seed=3930,
        reporting_delay=5,
        virus_shares=virus_shares,
        synthetic_data_path=paths["initial_states"],
        reported_infections_path=paths["rki"],
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

            The dictionary has the following entries:
                - initial_states
                - output_of_check_initial_states
                - contact_models
                - contact_policies
                - testing
                - initial_conditions
                - initial_infections
                - initial_immunity
                - susceptibility_factor_model
                - virus_shares
                - vaccination_models
                - vaccination_shares
                - rapid_test_models
                - rapid_test_reaction_models
                - seasonality_factor_model
                - params

    """
    if debug:
        initial_states_path = BLD / "data" / "debug_initial_states.parquet"
    else:
        initial_states_path = BLD / "data" / "initial_states.parquet"

    out = {
        "initial_states": initial_states_path,
        # to ensure that the checks on the initial states run before the
        # simulations we add the output of task_check_initial_states here
        # even though we don't use it.
        "output_of_check_initial_states": BLD
        / "data"
        / "comparison_of_age_group_distributions.png",
        "contact_models": SRC / "contact_models" / "get_contact_models.py",
        "contact_policies": SRC / "policies" / "enacted_policies.py",
        "testing": SRC / "testing" / "testing_models.py",
        "share_of_tests_for_symptomatics_series": BLD
        / "data"
        / "testing"
        / "share_of_tests_for_symptomatics_series.pkl",
        "initial_conditions": SRC
        / "create_initial_states"
        / "create_initial_conditions.py",
        "initial_infections": SRC
        / "create_initial_states"
        / "create_initial_infections.py",
        "initial_immunity": SRC
        / "create_initial_states"
        / "create_initial_immunity.py",
        "susceptibility_factor_model": SRC
        / "simulation"
        / "calculate_susceptibility.py",
        "virus_shares": BLD / "data" / "virus_strains" / "virus_shares_dict.pkl",
        "vaccination_models": SRC / "policies" / "find_people_to_vaccinate.py",
        "vaccination_shares": BLD
        / "data"
        / "vaccinations"
        / "vaccination_shares_extended.pkl",
        "rapid_test_models": SRC / "testing" / "rapid_tests.py",
        "rapid_test_reaction_models": SRC / "testing" / "rapid_test_reactions.py",
        "seasonality_factor_model": SRC / "simulation" / "seasonality.py",
        "params": BLD / "params.pkl",
        "scenario_simulation_inputs": SRC
        / "simulation"
        / "scenario_simulation_inputs.py",
        "params_scenarios": SRC / "simulation" / "params_scenarios.py",
        "rki": BLD / "data" / "processed_time_series" / "rki.pkl",
    }

    return out


def named_scenarios_to_parametrization(named_scenarios, fast_flag):
    """Convert named scenarios to parametrization.

    Each named scenario is duplicated with different seeds to capture the uncertainty in
    the simulation..

    """
    scenarios = []
    for name, specs in named_scenarios.items():
        for seed in range(specs["n_seeds"]):
            produces = create_path_to_last_states_of_simulation(fast_flag, name, seed)
            scaled_seed = 500 + 100_000 * seed
            spec_tuple = (
                specs["sim_input_scenario"],
                specs["params_scenario"],
                specs["start_date"],
                specs["end_date"],
                produces,
                scaled_seed,
            )
            scenarios.append(spec_tuple)

    return scenarios


def create_path_to_last_states_of_simulation(fast_flag, name, seed):
    return Path(
        BLD
        / "simulations"
        / f"{fast_flag}_{name}_{seed}"
        / "last_states"
        / "last_states.parquet"
    )
