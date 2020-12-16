from functools import partial

import pandas as pd

import src.policies.full_policy_blocks as fpb
from src.policies.domain_level_policy_blocks import _get_base_policy
from src.policies.policy_tools import combine_dictionaries
from src.policies.single_policy_functions import (
    reduce_contacts_through_private_contact_tracing,
)


def get_estimation_policies(contact_models):
    """Policies from 2020-04-23 to 2020-12-15."""
    reopening_end_multipliers = {"educ": 0.8, "work": 0.6, "other": 0.7}
    to_combine = [
        # Spring and Summer
        fpb.get_german_reopening_phase(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-04-23",
                "end_date": "2020-09-30",
                "prefix": "reopening",
            },
            start_multipliers={"educ": 0.5, "work": 0.2, "other": 0.3},
            end_multipliers=reopening_end_multipliers,
            educ_switching_date="2020-08-01",
        ),
        # October Policies
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-01",
                "end_date": "2020-10-20",
                "prefix": "after_reopening",
            },
            multipliers=reopening_end_multipliers,
        ),
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-21",
                "end_date": "2020-11-01",
                "prefix": "anticipate_lockdown_light",
            },
            multipliers={"educ": 0.8, "work": 0.6, "other": 0.55},
        ),
        # Light Lockdown from November to mid of December
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-02",
                "end_date": "2020-12-15",
                "prefix": "lockdown_light",
            },
            multipliers={"educ": 0.7, "work": 0.5, "other": 0.45},
        ),
    ]

    return combine_dictionaries(to_combine)


def get_december_to_feb_policies(
    contact_models,
    contact_tracing_multiplier,
    pre_christmas_other_multiplier=0.3,
    christmas_other_multiplier=0.0,
    post_christmas_multiplier=0.3,
):
    """Get policies from December 2020 to February 2021.

    Args:
        contact_models (dict): sid contact model dictionary.
        pre_christmas_other_multiplier (float): Factor

    """
    reopening_end_multipliers = {"educ": 0.8, "work": 0.6, "other": 0.7}
    to_combine = [
        # 1st December Half
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-01",
                "end_date": "2020-12-15",
                "prefix": "lockdown_light",
            },
            multipliers={"educ": 0.7, "work": 0.5, "other": 0.45},
        ),
        # Until Christmas
        fpb.get_hard_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-16",
                "end_date": "2020-12-23",
                "prefix": "pre-christmas-lockdown",
            },
            # use the start multiplier of reopening
            other_contacts_multiplier=pre_christmas_other_multiplier,
        ),
        # Christmas Holidays
        fpb.get_hard_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-24",
                "end_date": "2020-12-26",
                "prefix": "christmas-lockdown",
            },
            # use the start multiplier of reopening
            other_contacts_multiplier=christmas_other_multiplier,
        ),
        # Christmas Until End of Hard Lockdown
        fpb.get_hard_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-27",
                "end_date": "2021-01-10",
                "prefix": "post-christmas-lockdown",
            },
            # use the start multiplier of reopening
            other_contacts_multiplier=post_christmas_multiplier,
        ),
        # Return to Reopened After Hard Lockdown
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-11",
                "end_date": "2021-01-31",
                "prefix": "reopened-january",
            },
            multipliers=reopening_end_multipliers,
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
    # households, educ contact models and christmas models don't get adjustment
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
