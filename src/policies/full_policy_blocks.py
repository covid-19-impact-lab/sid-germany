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
from src.policies.domain_level_policy_blocks import implement_general_schooling_policy
from src.policies.domain_level_policy_blocks import reduce_other_models
from src.policies.domain_level_policy_blocks import reduce_work_models
from src.policies.domain_level_policy_blocks import reopen_educ_models
from src.policies.domain_level_policy_blocks import reopen_other_models
from src.policies.domain_level_policy_blocks import reopen_work_models
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


def get_lockdown_with_multipliers(
    contact_models,
    block_info,
    multipliers,
    a_b_educ_options=None,
    emergency_options=None,
):
    """Reduce all contact models except for households by multipliers.

    Args:
        multipliers (dict): Contains keys "educ", "work" and "other".
            The "educ" entry is only applied to the education models
            that are not in A/B mode.
        a_b_educ_options (dict): For every education type ("school", "preschool",
            "nursery") that has A/B schooling, add name of the type
            as key and the others_attend, the subgroup_query, and rhythm as
            key-value dict.
            Note to use the types (e.g. school) and not the contact models
            (e.g. educ_school_1) as keys.  multipliers["educ"] is not used on top
            of the supplied hygiene multiplier but only used for open education models
            Default is no A/B education.
        emergency_options (dict): For every education type ("school", "preschool",
            "nursery") that has emergency care, add name of the type as key
            and the hygiene_multiplier and always_attend_query as key-value dict.
            Note to use the modes (e.g. school) and not the contact models
            (e.g. educ_school_1) as keys. The other supplied multiplier is
            not used on top of the supplied hygiene multiplier.

    """
    if a_b_educ_options is None:
        a_b_educ_options = {}
    if emergency_options is None:
        emergency_options = {}
    educ_policies = implement_general_schooling_policy(
        contact_models=contact_models,
        block_info=block_info,
        a_b_educ_options=a_b_educ_options,
        emergency_options=emergency_options,
        other_educ_multiplier=multipliers["educ"],
    )
    work_policies = reduce_work_models(contact_models, block_info, multipliers["work"])
    if multipliers["other"] == 0.0:
        other_policies = shut_down_other_models(contact_models, block_info)
    elif multipliers["other"] < 1.0:
        other_policies = reduce_other_models(
            contact_models, block_info, multipliers["other"]
        )
    elif multipliers["other"] == 1:
        other_policies = {}
    else:
        raise ValueError("Only other multipliers <= 1 allowed.")

    to_combine = [educ_policies, work_policies, other_policies]
    policies = combine_dictionaries(to_combine)
    return policies
