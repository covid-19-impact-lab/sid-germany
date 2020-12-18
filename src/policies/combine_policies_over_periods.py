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
    pre_christmas_multiplier=0.4,
    christmas_other_multiplier=0.0,
    post_christmas_multiplier=0.4,
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
        pre_christmas_other_multiplier (float, optional):
            Other multiplier passed to the lockdown before Christmas.
        christmas_other_multiplier (float, optional):
            Other multiplier passed to the lockdown during Christmas.
        post_christmas_other_multiplier (float, optional):
            Other multiplier passed to the lockdown after Christmas.

    Returns:
        policies (dict): policies dictionary.

    """
    to_combine = [
        # 1st December Half
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-01",
                "end_date": "2020-12-15",
                "prefix": "lockdown_light",
            },
            multipliers={"educ": 0.7, "work": 0.55, "other": 0.4},
        ),
        # Until Christmas
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-16",
                "end_date": "2020-12-23",
                "prefix": "pre-christmas-lockdown",
            },
            multipliers={
                "educ": 0.0,
                "work": pre_christmas_multiplier,
                "other": pre_christmas_multiplier,
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
            other_contacts_multiplier=christmas_other_multiplier,
        ),
        # Christmas Until End of Hard Lockdown
        fpb.get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-27",
                "end_date": "2021-01-11",
                "prefix": "post-christmas-lockdown",
            },
            multipliers={
                "educ": 0.0,
                "work": post_christmas_multiplier,
                "other": post_christmas_multiplier,
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