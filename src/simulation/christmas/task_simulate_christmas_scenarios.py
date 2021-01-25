from functools import partial

import pandas as pd
import pytask
import sid

import src.policies.full_policy_blocks as fpb
from src.config import BLD
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (
    create_initial_conditions,
)
from src.policies.domain_level_policy_blocks import _get_base_policy
from src.policies.policy_tools import combine_dictionaries
from src.policies.single_policy_functions import (
    reduce_contacts_through_private_contact_tracing,
)
from src.simulation.christmas.spec_christmas_scenarios import (
    create_christmas_parametrization,
)


SIMULATION_START = pd.Timestamp("2020-12-02")
INITIAL_START = SIMULATION_START - pd.Timedelta(days=31)
SIMULATION_END = pd.Timestamp("2021-01-08")


@pytask.mark.skip
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
    }
)
@pytask.mark.parametrize(
    "scenario, christmas_mode, contact_tracing_multiplier, path, produces",
    create_christmas_parametrization(),
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


def get_december_to_feb_policies(
    contact_models,
    contact_tracing_multiplier,
    scenario,
    path=None,
):
    """Get policies from December 2020 to February 2021.

    Args:
        contact_models (dict): sid contact model dictionary.
        contact_tracing_multiplier (float, optional):
            If not None, private contact tracing takes place
            between the 27.12. and 10.01, i.e. in the two
            weeks after Christmas. The multiplier is the
            reduction multiplier for recurrent and non-recurrent
            contact models.
        scenario (str): One of "optimistic", "pessimistic"
        path (str or pathlib.Path): Path to a folder in which information on the
            contact tracing is stored.

    Returns:
        policies (dict): policies dictionary.

    """
    if scenario == "optimistic":
        hard_lockdown_work_multiplier = 0.33 + 0.66 * 0.3
        vacation_other_multiplier = 0.4
        hard_lockdown_other_multiplier = 0.3
    elif scenario == "pessimistic":
        hard_lockdown_work_multiplier = 0.33 + 0.66 * 0.4
        vacation_other_multiplier = 0.7
        hard_lockdown_other_multiplier = 0.4
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    to_combine = [
        # 1st December Half
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-01",
                "end_date": "2020-12-15",
                "prefix": "lockdown_light",
            },
            multipliers={"educ": 0.7, "work": 0.33 + 0.66 * 0.45, "other": 0.5},
        ),
        # Until start of christmas vacation
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-16",
                "end_date": "2020-12-20",
                "prefix": "pre-christmas-lockdown-first-half",
            },
            multipliers={
                "educ": 0.0,
                "work": hard_lockdown_work_multiplier,
                "other": hard_lockdown_other_multiplier,
            },
        ),
        # until christmas
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-21",
                "end_date": "2020-12-23",
                "prefix": "pre-christmas-lockdown-second-half",
            },
            multipliers={
                "educ": 0.0,
                "work": 0.33 + 0.66 * 0.15,
                "other": hard_lockdown_other_multiplier,
            },
        ),
        # Christmas Holidays
        fpb.get_hard_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-24",
                "end_date": "2020-12-26",
                "prefix": "christmas-lockdown",
            },
            other_contacts_multiplier=0.2,
        ),
        # Christmas Until End of Hard Lockdown
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-27",
                "end_date": "2021-01-03",
                "prefix": "post-christmas-lockdown",
            },
            multipliers={
                "educ": 0.0,
                "work": 0.33 + 0.66 * 0.15,
                "other": vacation_other_multiplier,
            },
        ),
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-04",
                "end_date": "2021-01-11",
                "prefix": "after-christmas-vacation",
            },
            multipliers={
                "educ": 0.0,
                "work": hard_lockdown_work_multiplier,
                "other": hard_lockdown_other_multiplier,
            },
        ),
    ]
    if contact_tracing_multiplier is not None:
        contact_tracing_policies = _get_christmas_contact_tracing_policies(
            contact_models=contact_models,
            block_info={
                "start_date": pd.Timestamp("2020-12-27"),
                "end_date": pd.Timestamp("2021-01-10"),
                "prefix": "private-contact-tracing",
            },
            multiplier=contact_tracing_multiplier,
            path=path,
        )
        to_combine.append(contact_tracing_policies)

    return combine_dictionaries(to_combine)


def _get_christmas_contact_tracing_policies(
    contact_models, block_info, multiplier, path=None
):
    """"""
    # households, educ contact models and Christmas models don't get adjustment
    models_with_post_christmas_isolation = [
        cm for cm in contact_models if "work" in cm or "other" in cm
    ]
    christmas_id_groups = list(
        {
            model["assort_by"][0]
            for name, model in contact_models.items()
            if "christmas" in name
        }
    )
    policies = {}
    for mod in models_with_post_christmas_isolation:
        policy = _get_base_policy(mod, block_info)
        policy["policy"] = partial(
            reduce_contacts_through_private_contact_tracing,
            multiplier=multiplier,
            group_ids=christmas_id_groups,
            is_recurrent=contact_models[mod]["is_recurrent"],
            path=path,
        )
        policies[f"{block_info['prefix']}_{mod}"] = policy
    return policies
