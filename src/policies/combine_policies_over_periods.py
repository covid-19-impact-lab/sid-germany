from functools import partial

import pandas as pd

import src.policies.full_policy_blocks as fpb
from src.policies.domain_level_policy_blocks import _get_base_policy
from src.policies.policy_tools import combine_dictionaries
from src.policies.single_policy_functions import (
    reduce_contacts_through_private_contact_tracing,
)


def get_december_to_feb_policies(
    contact_models,
    contact_tracing_multiplier,
    scenario,
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
    Returns:
        policies (dict): policies dictionary.
    """
    if scenario == "optimistic":
        hard_lockdown_work_multiplier = 0.35
        vacation_other_multiplier = 0.4
        hard_lockdown_other_multiplier = 0.35
    elif scenario == "pessimistic":
        hard_lockdown_work_multiplier = 0.4
        vacation_other_multiplier = 0.8
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
        contact_tracing_policies = get_christmas_contact_tracing_policies(
            contact_models=contact_models,
            block_info={
                "start_date": pd.Timestamp("2020-12-27"),
                "end_date": pd.Timestamp("2021-01-10"),
                "prefix": "private-contact-tracing",
            },
            multiplier=contact_tracing_multiplier,
        )
        to_combine.append(contact_tracing_policies)
    return combine_dictionaries(to_combine)


def get_christmas_contact_tracing_policies(contact_models, block_info, multiplier):
    """"""
    # households, educ contact models and Christmas models don't get adjustment
    models_with_post_christmas_isolation = [
        cm for cm in contact_models if "work" in cm or "other" in cm
    ]
    christmas_id_groups = [
        model["assort_by"][0]
        for name, model in contact_models.items()
        if "christmas" in name
    ]
    policies = {}
    for mod in models_with_post_christmas_isolation:
        policy = _get_base_policy(mod, block_info)
        policy["policy"] = partial(
            reduce_contacts_through_private_contact_tracing,
            multiplier=multiplier,
            group_ids=christmas_id_groups,
            is_recurrent=contact_models[mod]["is_recurrent"],
        )
        policies[f"{block_info['prefix']}_{mod}"] = policy
    return policies
