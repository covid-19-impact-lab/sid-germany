"""Combine policies to blocks with common start and end date.

These functions deliberately are not as flexible and abstract as they could be
in order to make them less error prone. If you need something that is not here,
consider adding another function before you make an existing one more flexible.

All functions take an argument called ``block_info`` that contains the start_date,
end_date and prefix of the policy block. The prefix is used to construct the
policy names.

Moreover, all functions take contact_models as argument which is a sid compatible
cotact model dictionary. Some of them, additionally let the user set the strictness of
policies.

Policy strictness is always set in terms of a multiplier, i.e a number between zero and
one that enters multiplicatively to reduce something.

Alternatives to multipliers would be reduction factors, but then it is not clear whether
this is defined as (1 - multiplier) or (1 / multiplier) which makes them error prone.
Thus we do not use anything but multipliers here!

"""
from src.policies.domain_level_policy_blocks import (
    implement_ab_schooling_with_reduced_other_educ_models,
)
from src.policies.domain_level_policy_blocks import reduce_educ_models
from src.policies.domain_level_policy_blocks import reduce_other_models
from src.policies.domain_level_policy_blocks import reduce_work_models
from src.policies.domain_level_policy_blocks import reopen_educ_models
from src.policies.domain_level_policy_blocks import reopen_other_models
from src.policies.domain_level_policy_blocks import reopen_work_models
from src.policies.domain_level_policy_blocks import shut_down_educ_models
from src.policies.domain_level_policy_blocks import shut_down_other_models
from src.policies.policy_tools import combine_dictionaries


def get_german_reopening_phase(
    contact_models,
    block_info,
    start_multipliers,
    end_multipliers,
    educ_switching_date="2020-08-01",
    educ_reopening_dates=None,
):
    """Model German summer 2020 as graudal reopening of schools, economy and leisure.

    This function is very specific and not meant to be used in any other context than
    to replicate actual German policies from April to September during estimation.

    Args:
        start_multipliers (dict): Dictionary with the following entries:
            - "educ": Multiplier for activity in educ models after reopening
                but before summer vacation.
            - "work": Multiplier for work contacts at beginning of reopening phase.
                Note that essential workers always work. Thus the multiplier is the
                share of active non-essential workers.
            - "other": Multiplier for non-work and non-educ contacts at beginning of
                reopening phase.
        end_multipliers (dict): Dictionary with same entries as start_multipliers but
            for activity levels at the end of the reopening phase.
        educ_switching_date (str or pandas.Timestamp): Date at which the multiplier
            switches from start to end multiplier in education models.
        educ_reopening_dates (dict): Maps German federal states to reopening dates
            for education models.

    """
    to_combine = [
        reopen_educ_models(
            contact_models=contact_models,
            block_info=block_info,
            start_multiplier=start_multipliers["educ"],
            end_multiplier=end_multipliers["educ"],
            switching_date=educ_switching_date,
            reopening_dates=educ_reopening_dates,
        ),
        reopen_work_models(
            contact_models=contact_models,
            block_info=block_info,
            start_multiplier=start_multipliers["work"],
            end_multiplier=end_multipliers["work"],
        ),
        reopen_other_models(
            contact_models=contact_models,
            block_info=block_info,
            start_multiplier=start_multipliers["other"],
            end_multiplier=end_multipliers["other"],
        ),
    ]

    policies = combine_dictionaries(to_combine)
    return policies


def get_lockdown_with_multipliers(contact_models, block_info, multipliers):
    """Reduce all contact models except for households by multipliers."""
    if multipliers["educ"] == 0.0:
        educ_policies = shut_down_educ_models(contact_models, block_info)
    elif multipliers["educ"] < 1.0:
        educ_policies = reduce_educ_models(
            contact_models, block_info, multipliers["educ"]
        )
    else:
        educ_policies = {}

    work_policies = reduce_work_models(contact_models, block_info, multipliers["work"])
    if multipliers["other"] == 0.0:
        other_policies = shut_down_other_models(contact_models, block_info)
    elif multipliers["other"] < 1.0:
        other_policies = reduce_other_models(
            contact_models, block_info, multipliers["other"]
        )
    else:
        other_policies = {}
    to_combine = [educ_policies, work_policies, other_policies]
    policies = combine_dictionaries(to_combine)
    return policies


def get_lockdown_with_multipliers_with_ab_schooling(
    contact_models, block_info, multipliers, age_cutoff
):

    to_combine = [
        implement_ab_schooling_with_reduced_other_educ_models(
            contact_models=contact_models,
            block_info=block_info,
            age_cutoff=age_cutoff,
            multiplier=multipliers["educ"],
        ),
        reduce_work_models(contact_models, block_info, multipliers["work"]),
        reduce_other_models(contact_models, block_info, multipliers["other"]),
    ]

    policies = combine_dictionaries(to_combine)
    return policies
