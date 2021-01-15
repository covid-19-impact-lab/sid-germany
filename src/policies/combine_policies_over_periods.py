from functools import partial

import pandas as pd

import src.policies.full_policy_blocks as fpb
from src.policies.domain_level_policy_blocks import _get_base_policy
from src.policies.full_policy_blocks import get_german_reopening_phase
from src.policies.full_policy_blocks import get_soft_lockdown
from src.policies.policy_tools import combine_dictionaries
from src.policies.single_policy_functions import (
    reduce_contacts_through_private_contact_tracing,
)


def get_estimation_policies(contact_models):
    """Get policies from July to December 20th."""
    reopening_start_multipliers = {"educ": 0.8, "work": 0.55, "other": 0.45}
    reopening_end_multipliers = {"educ": 0.8, "work": 0.95, "other": 0.7}
    anticipate_lockdown_multipliers = {"educ": 0.8, "work": 0.6, "other": 0.5}
    lockdown_light_multipliers = {"educ": 0.6, "work": 0.45, "other": 0.4}
    ### NOT USED IN FITNESS PLOTS NOTEBOOK!
    lockdown_light_multipliers_with_fatigue = {"educ": 0.6, "work": 0.45, "other": 0.5}
    to_combine = [
        get_german_reopening_phase(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-07-01",
                "end_date": "2020-10-25",
                "prefix": "reopening",
            },
            start_multipliers=reopening_start_multipliers,
            end_multipliers=reopening_end_multipliers,
            educ_switching_date="2020-08-01",
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-26",
                "end_date": "2020-11-01",
                "prefix": "anticipate_lockdown_light",
            },
            multipliers=anticipate_lockdown_multipliers,
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-02",
                "end_date": "2020-11-22",
                "prefix": "lockdown_light",
            },
            multipliers=lockdown_light_multipliers,
        ),
        ### THE FOLLOWING IS DIFFERENT FROM OUR CHRISTMAS SCENARIO POLICIES!
        ### -> ADJUST ONE / THE OTHER?
        ### -> PROBABLY SHOULDN'T OVERLAP
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-23",
                "end_date": "2020-12-20",
                "prefix": "lockdown_light_with_fatigue",
            },
            multipliers=lockdown_light_multipliers,
        ),
    ]

    return combine_dictionaries(to_combine)


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
        hard_lockdown_work_multiplier = 0.3
        vacation_other_multiplier = 0.4
        hard_lockdown_other_multiplier = 0.3
    elif scenario == "pessimistic":
        hard_lockdown_work_multiplier = 0.4
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
            multipliers={"educ": 0.7, "work": 0.45, "other": 0.5},
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
                "work": 0.15,
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
                "work": 0.15,
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
