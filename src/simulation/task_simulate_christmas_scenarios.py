import pandas as pd
import pytask
import sid

from src.config import BLD
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (
    create_initial_conditions,
)
from src.policies.combine_policies_over_periods import get_december_to_feb_policies
from src.simulation.spec_christmas_scenarios import CARTESIAN_PRODUCT
from src.simulation.spec_christmas_scenarios import create_output_path_for_simulation
from src.simulation.spec_christmas_scenarios import create_path_to_last_states

SIMULATION_START = pd.Timestamp("2020-12-02")
INITIAL_START = SIMULATION_START - pd.Timedelta(days=31)
SIMULATION_END = pd.Timestamp("2021-01-08")


def _create_christmas_parametrization():
    """Create the parametrization for the simulation of the Christmas scenarios.

    Returns:
        out (list): List of specification tuples. Each tuple is composed of:
            (scenario, christmas_mode, contact_tracing_multiplier, path, produces).
            path is the directory where sid saves all results.
            produces is the path to the last states.

    """
    paths = [create_output_path_for_simulation(*args) for args in CARTESIAN_PRODUCT]
    produces = [create_path_to_last_states(*args) for args in CARTESIAN_PRODUCT]

    return zip(*zip(*CARTESIAN_PRODUCT), paths, produces)


@pytask.mark.persist
@pytask.mark.depends_on(
    {
        "params": BLD / "start_params.pkl",
        "estimated_params": SRC / "simulation" / "estimated_params.pkl",
        "share_known_cases": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases.pkl",
        "initial_states": BLD / "data" / "initial_states.parquet",
        "contact_model_functions": SRC
        / "contact_models"
        / "contact_model_functions.py",
        "get_contact_models": SRC / "contact_models" / "get_contact_models.py",
        "get_policies": SRC / "policies" / "combine_policies_over_periods.py",
    }
)
@pytask.mark.parametrize(
    "scenario, christmas_mode, contact_tracing_multiplier, path, produces",
    _create_christmas_parametrization(),
)
def task_simulation_christmas_scenarios(
    depends_on, scenario, christmas_mode, contact_tracing_multiplier, path
):

    share_known_cases = pd.read_pickle(depends_on["share_known_cases"])

    df = pd.read_parquet(depends_on["initial_states"])

    # assume 5% 1 contact. 5% 2 contacts before the holidays (21 to 23rd of Dec)
    contact_dist = pd.Series(data=[0.9, 0.05, 0.05], index=[0, 1, 2])
    contact_models = get_all_contact_models(
        christmas_mode=christmas_mode, n_extra_contacts_before_christmas=contact_dist
    )

    # update with the estimated params
    params = pd.read_pickle(depends_on["params"])
    estimated_params = pd.read_pickle(depends_on["estimated_params"])
    rows_to_overwrite = [x for x in estimated_params.index if x in params.index]
    params.loc[rows_to_overwrite] = estimated_params.loc[rows_to_overwrite]
    rows_to_add = [x for x in estimated_params.index if x not in params.index]
    params = pd.concat([params, estimated_params.loc[rows_to_add]])
    # add christmas infection probs by hand
    other_non_rec_loc = ("infection_prob", "other_non_recurrent", "other_non_recurrent")
    holiday_prep_loc = ("infection_prob", "holiday_preparation", "holiday_preparation")
    params.loc[holiday_prep_loc] = params.loc[other_non_rec_loc]
    missing_christmas_models = []
    for x in contact_models:
        if x not in params.loc["infection_prob"].index.get_level_values("subcategory"):
            missing_christmas_models.append(x)
    hh_loc = ("infection_prob", "households", "households")
    for name in missing_christmas_models:
        params.loc[("infection_prob", name, name)] = 1.5 * params.loc[hh_loc]

    policies = get_december_to_feb_policies(
        contact_models=contact_models,
        contact_tracing_multiplier=contact_tracing_multiplier,
        scenario=scenario,
        path=path,
    )

    initial_conditions = create_initial_conditions(
        start=INITIAL_START,
        end=SIMULATION_START - pd.Timedelta(days=1),
        reporting_delay=5,
        seed=99,
    )

    simulate = sid.get_simulate_func(
        params=params,
        initial_states=df,
        contact_models=contact_models,
        duration={"start": SIMULATION_START, "end": SIMULATION_END},
        contact_policies=policies,
        initial_conditions=initial_conditions,
        share_known_cases=share_known_cases,
        path=path,
        events=None,
        seed=384,
        saved_columns={
            "time": ["date"],
            "initial_states": False,
            "disease_states": ["symptomatic", "ever_infected", "newly_infected"],
            "other": ["n_has_infected", "new_known_case"],
        },
    )

    simulate(params)
