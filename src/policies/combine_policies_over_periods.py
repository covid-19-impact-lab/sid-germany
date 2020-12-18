from functools import partial

import pandas as pd

from src.policies.domain_level_policy_blocks import _get_base_policy
from src.policies.full_policy_blocks import get_hard_lockdown
from src.policies.full_policy_blocks import get_soft_lockdown
from src.policies.policy_tools import combine_dictionaries
from src.policies.single_policy_functions import (
    reduce_contacts_through_private_contact_tracing,
)


def get_december_to_feb_policies(
    contact_models,
    contact_tracing_multiplier,
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

    Returns:
        policies (dict): policies dictionary.

    """
    november_multiplier = {"educ": 0.7, "work": 0.55, "other": 0.4}
    hard_lockdown_multiplier = {"educ": 0.0, "work": 0.4, "other": 0.4}
    to_combine = [
        # 1st December Half
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-01",
                "end_date": "2020-12-15",
                "prefix": "lockdown-light",
            },
            multipliers=november_multiplier,
        ),
        # 16.12. - 20.12.
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-16",
                "end_date": "2020-12-20",
                "prefix": "pre-christmas-lockdown",
            },
            multipliers=hard_lockdown_multiplier,
        ),
        # 21.12. - 03.01.
        get_hard_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-21",
                "end_date": "2021-01-02",
                "prefix": "christmas-lockdown",
            },
            other_contacts_multiplier=0.8,
        ),
        # Until End of Hard Lockdown
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-01-03",
                "end_date": "2021-01-10",
                "prefix": "post-christmas-lockdown",
            },
            multipliers=hard_lockdown_multiplier,
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
