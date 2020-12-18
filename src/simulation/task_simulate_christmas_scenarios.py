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


SIMULATION_START = pd.Timestamp("2020-12-02")
INITIAL_START = SIMULATION_START - pd.Timedelta(days=31)
SIMULATION_END = pd.Timestamp("2021-01-10")


def create_christmas_parametrization():
    parametrizations = []
    for christmas_mode in ["full", "same_group", "meet_twice"]:
        for contact_tracing_multiplier in [None, 0.5, 0.1]:
            ctm_str = (
                "wo_ct"
                if contact_tracing_multiplier is None
                else f"w_ct_{str(contact_tracing_multiplier).replace('.', '_')}"
            )
            path = (
                BLD / "simulation" / f"simulation_christmas_mode_{christmas_mode}_"
                f"{ctm_str}"
            )
            produces = path / "time_series"

            single_run = (christmas_mode, contact_tracing_multiplier, path, produces)

            parametrizations.append(single_run)

    return parametrizations


PARAMETRIZATIONS = create_christmas_parametrization()


@pytask.mark.depends_on(
    {
        "params": BLD / "start_params.pkl",
        "estimated_params": SRC / "simulation" / "best_so_far.pkl",
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
    "christmas_mode, contact_tracing_multiplier, path, produces", PARAMETRIZATIONS
)
def task_simulation_christmas_scenarios(
    depends_on, christmas_mode, contact_tracing_multiplier, path
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
        params.loc[("infection_prob", name, name)] = params.loc[hh_loc]

    policies = get_december_to_feb_policies(
        contact_models=contact_models,
        pre_christmas_multiplier=0.4,
        christmas_other_multiplier=0.0,
        post_christmas_multiplier=0.4,
        contact_tracing_multiplier=contact_tracing_multiplier,
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
            "disease_states": ["symptomatic"],
            "other": ["n_has_infected", "newly_infected", "new_known_case"],
        },
    )

    simulate(params)