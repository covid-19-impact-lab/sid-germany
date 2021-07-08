import warnings
from functools import partial

import pandas as pd
from sid.statistics import calculate_r_effective

from src.calculate_moments import calculate_period_outcome_sim
from src.config import BLD
from src.config import SID_DEPENDENCIES
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (
    create_initial_conditions,
)
from src.events.events import introduce_b117
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
    return_last_states=False,
    initial_states_path=None,
    is_resumed=False,
    rapid_test_statistics_path=None,
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
        return_last_states (bool, optional): if True, the last states are returned as
            part of the simulation result.
        initial_states_path (pathlib.Path, optional): Path to the initial states.
            If not given the standard initial states are used.
        is_resumed (bool, optional): if True, the initial_states_path must be given. In
            that case no initial conditions are created
        rapid_test_statistics_path (Path, optional): where to save rapid test
            statistics.


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
    if is_resumed:
        assert (
            initial_states_path is not None
        ), "You must specify the path to the initial states if you resume a simulation."

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    paths = get_simulation_dependencies(
        debug=debug,
        is_resumed=is_resumed,
    )
    if rapid_test_statistics_path is not None:
        paths["rapid_test_statistics_path"] = rapid_test_statistics_path

    if initial_states_path is None:
        initial_states_path = paths["initial_states"]

    if initial_states_path.suffix == ".pkl":
        initial_states = pd.read_pickle(initial_states_path)
    elif initial_states_path.suffix == ".parquet":
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

    if not is_resumed:
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
    else:
        initial_conditions = None

    seasonality_factor_model = partial(seasonality_model, contact_models=contact_models)

    def _currently_infected(df):
        return df["infectious"] | df["symptomatic"] | (df["cd_infectious_true"] >= 0)

    def _knows_currently_infected(df):
        return df["knows_immune"] & df["currently_infected"]

    derived_state_variables = {
        "currently_infected": _currently_infected,
        "knows_currently_infected": _knows_currently_infected,
    }

    events = {"introduce_b117": {"model": introduce_b117}}

    fixed_inputs = {
        "initial_states": initial_states,
        "contact_models": contact_models,
        "duration": duration,
        "events": events,
        "testing_demand_models": testing_demand_models,
        "testing_allocation_models": testing_allocation_models,
        "testing_processing_models": testing_processing_models,
        "saved_columns": saved_columns,
        "initial_conditions": initial_conditions,
        "susceptibility_factor_model": calculate_susceptibility,
        "virus_strains": ["base_strain", "b117"],
        "seasonality_factor_model": seasonality_factor_model,
        "derived_state_variables": derived_state_variables,
        "return_last_states": return_last_states,
    }

    if period_outputs:
        fixed_inputs["period_outputs"] = create_period_outputs()
        fixed_inputs["return_time_series"] = False

    scenario_func = getattr(scenario_simulation_inputs, scenario)
    scenario_inputs = scenario_func(paths, fixed_inputs)
    simulation_inputs = combine_dictionaries([fixed_inputs, scenario_inputs])
    return simulation_inputs


def get_simulation_dependencies(debug, is_resumed):
    """Collect paths on which the simulation depends.

    This contains both paths to python modules and data paths.
    It only covers sid-germany specific paths, i.e. not sid.

    Args:
        debug (bool): Whether to use the debug initial states.
        is_resumed (bool): Whether the simulation is a resumed simulation. If False the
            path to the initial states from BLD / "data" are given.

    Returns:
        paths (dict): Dictionary with the dependencies for the simulation.

    """
    out = {
        **SID_DEPENDENCIES,
        # to ensure that the checks on the initial states run before the
        # simulations we add the output of task_check_initial_states here
        # even though we don't use it.
        "output_of_check_initial_states": BLD
        / "figures"
        / "data"
        / "how_well_our_synthetic_population_matches_the_german_age_distribution.pdf",
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
        "create_rapid_test_statistics": SRC
        / "testing"
        / "create_rapid_test_statistics.py",
    }

    if not is_resumed:
        if debug:
            out["initial_states"] = BLD / "data" / "debug_initial_states.parquet"
        else:
            out["initial_states"] = BLD / "data" / "initial_states.parquet"

    return out


def create_period_outputs():
    period_outputs = {}

    incidence_outcomes = [
        "newly_infected",
        "new_known_case",
        "newly_deceased",
        "currently_infected",  # only used for share known cases
        "knows_currently_infected",  # only used for share known cases
        "ever_vaccinated",
    ]
    groupbys = ["state", "age_group_rki", None]

    for outcome in incidence_outcomes:
        for groupby in groupbys:
            gb_str = f"_by_{groupby}" if groupby is not None else ""
            period_outputs[outcome + gb_str] = partial(
                calculate_period_outcome_sim,
                outcome=outcome,
                groupby=groupby,
            )

    period_outputs["r_effective"] = partial(calculate_r_effective, window_length=7)
    period_outputs["share_b117"] = calculate_period_virus_share

    for groupby in groupbys:
        gb_str = f"_by_{groupby}" if groupby is not None else ""
        period_outputs["share_ever_rapid_test" + gb_str] = partial(
            _calculate_share_ever_rapid_test, groupby=groupby
        )
        period_outputs["share_rapid_test_in_last_week" + gb_str] = partial(
            _calculate_share_rapid_test_in_last_week, groupby=groupby
        )
        period_outputs["share_doing_rapid_test_today" + gb_str] = partial(
            _calculate_share_doing_rapid_test, groupby=groupby
        )

    return period_outputs


def _calculate_share_ever_rapid_test(df, groupby):
    sid_start_value = -9999
    ever_tested_share_per_group = _calculate_share_rapid_test_countdown_between(
        df=df,
        groupby=groupby,
        lower=sid_start_value + 1,
        upper=0,
        name="ever_had_a_rapid_test",
    )
    return ever_tested_share_per_group


def _calculate_share_rapid_test_in_last_week(df, groupby):
    tested_last_week = _calculate_share_rapid_test_countdown_between(
        df=df,
        groupby=groupby,
        lower=-6,
        upper=0,
        name="last_rapid_test_in_the_last_week",
    )
    return tested_last_week


def _calculate_share_rapid_test_countdown_between(df, groupby, lower, upper, name):
    """Share by groupby with their last rapid test between lower and upper (inclusively).

    Args:
        df (pandas.DataFrame): states DataFrame
        groupby (str or None): groupby column
        lower (int): lower bound (individuals with this value are included)
        upper (int): upper bound (individuals with this value are included)
        name (str): name of the Series to be returned

    """
    if groupby is None:
        groupby = []
    elif isinstance(groupby, str):
        groupby = [groupby]

    within_interval = df["cd_received_rapid_test"].between(lower, upper, inclusive=True)
    within_interval = within_interval.to_frame(name=name)
    within_interval["date"] = df["date"]
    if groupby:
        within_interval[groupby] = df[groupby]

    grouper = [pd.Grouper(key="date", freq="D")] + groupby
    within_interval_share_per_group = within_interval.groupby(grouper)[name].mean()

    return within_interval_share_per_group


def calculate_period_virus_share(df):
    df = df[["virus_strain", "date", "newly_infected"]]
    df = df[df["newly_infected"]]
    df["b117"] = df["virus_strain"] == "b117"

    out = df.groupby([pd.Grouper(key="date", freq="D")])["b117"].mean().fillna(0)
    return out


def _calculate_share_doing_rapid_test(df, groupby):
    if groupby is None:
        groupby = []
    elif isinstance(groupby, str):
        groupby = [groupby]

    df = df[["date", "cd_received_rapid_test"] + groupby].copy(deep=True)
    # has to be -1 because period outputs are done after countdowns are updated.
    df["did_rapid_test_today"] = df["cd_received_rapid_test"] == -1

    out = (
        df.groupby([pd.Grouper(key="date", freq="D")] + groupby)["did_rapid_test_today"]
        .mean()
        .fillna(0)
    )
    return out
