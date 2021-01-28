"""Functions to create building blocks of policies with common dates and domains.

Currently we distinguish the domains "work", "educ" and "other". "households" are not
included in any domain because they are currently not subject to policies.

The functions here should not be called directly but will be used in the module
"full_policy_blocks.py".

All public functions here take the following arguments (and possibly some more)

contact_models (dict): A sid compatible dictionary with contact models.
block_info (dict): A dictionary containing start_date, end_date and prefix of a
    block of policies.

The functions here expect that the domain names are part of contact model names.

"""
from functools import partial

from src.policies.single_policy_functions import implement_a_b_school_system_above_age
from src.policies.single_policy_functions import reduce_recurrent_model
from src.policies.single_policy_functions import reduce_work_model
from src.policies.single_policy_functions import reopen_educ_model_germany
from src.policies.single_policy_functions import reopen_other_model
from src.policies.single_policy_functions import reopen_work_model
from src.policies.single_policy_functions import shut_down_model

# ======================================================================================
# work policies
# ======================================================================================


def reduce_work_models(contact_models, block_info, multiplier):
    """Reduce contacts of workers by a multiplier."""
    policies = {}
    work_models = _get_work_models(contact_models)
    for mod in work_models:
        policy = _get_base_policy(mod, block_info)
        policy["policy"] = partial(reduce_work_model, multiplier=multiplier)
        policies[f"{block_info['prefix']}_{mod}"] = policy
    return policies


def reopen_work_models(contact_models, block_info, start_multiplier, end_multiplier):
    """Reduce contacts of workers with a gradually changing multiplier."""
    policies = {}
    work_models = _get_work_models(contact_models)
    for mod in work_models:
        policy = _get_base_policy(mod, block_info)
        policy["policy"] = partial(
            reopen_work_model,
            start_date=block_info["start_date"],
            end_date=block_info["end_date"],
            start_multiplier=start_multiplier,
            end_multiplier=end_multiplier,
        )
        policies[f"{block_info['prefix']}_{mod}"] = policy
    return policies


# ======================================================================================
# educ policies
# ======================================================================================


def shut_down_educ_models(contact_models, block_info):
    """Shut down all educ models to zero."""
    policies = {}
    educ_models = _get_educ_models(contact_models)
    for mod in educ_models:
        policy = _get_base_policy(mod, block_info)
        policy["policy"] = shut_down_model
        policies[f"{block_info['prefix']}_{mod}"] = policy
    return policies


def reduce_educ_models(contact_models, block_info, multiplier):
    """Reduce contacts in educ models with multiplier."""
    policies = {}
    educ_models = _get_educ_models(contact_models)
    for mod in educ_models:
        policy = _get_base_policy(mod, block_info)
        # currently all educ models are recurrent but don't want to assume it
        if contact_models[mod]["is_recurrent"]:
            policy["policy"] = partial(reduce_recurrent_model, multiplier=multiplier)
        else:
            policy = multiplier

        policies[f"{block_info['prefix']}_{mod}"] = policy
    return policies


def reopen_educ_models(
    contact_models,
    block_info,
    start_multiplier,
    end_multiplier,
    switching_date,
    reopening_dates,
):
    """Reopen an educ model at state specific dates

    - Keep the model closed until local reopening date
    - Work with strongly reduced contacts until summer vacation
    - Work with slightly reduced contact after summer vacation

    """
    policies = {}
    educ_models = _get_educ_models(contact_models)
    for mod in educ_models:
        policy = _get_base_policy(mod, block_info)
        policy["policy"] = partial(
            reopen_educ_model_germany,
            start_multiplier=start_multiplier,
            end_multiplier=end_multiplier,
            switching_date=switching_date,
            reopening_dates=reopening_dates,
            is_recurrent=contact_models[mod]["is_recurrent"],
        )
        policies[f"{block_info['prefix']}_{mod}"] = policy
    return policies


def implement_ab_schooling_with_reduced_other_educ_models(
    contact_models,
    block_info,
    age_cutoff,
    multiplier,
):
    """Split classes on a weekly basis for schools; keep other educ models open."""
    policies = reduce_educ_models(contact_models, block_info, multiplier)

    educ_models = _get_educ_models(contact_models)
    school_models = [
        cm for cm in educ_models if "school" in cm and "preschool" not in cm
    ]

    for mod in school_models:
        pol_name = f"{block_info['prefix']}_{mod}"
        assert pol_name in policies

        new_policy = _get_base_policy(mod, block_info)
        new_policy["policy"] = partial(
            implement_a_b_school_system_above_age,
            age_cutoff=age_cutoff,
        )
        policies[pol_name] = new_policy

    return policies


# ======================================================================================
# other policies
# ======================================================================================


def shut_down_other_models(contact_models, block_info):
    """Reduce all other contacts to zero."""
    policies = {}
    other_models = _get_other_models(contact_models)
    for mod in other_models:
        policy = _get_base_policy(mod, block_info)
        policy["policy"] = shut_down_model
        policies[f"{block_info['prefix']}_{mod}"] = policy
    return policies


def reduce_other_models(contact_models, block_info, multiplier):
    """Reduce other contacts with multiplier."""
    policies = {}
    other_models = _get_other_models(contact_models)
    for mod in other_models:
        policy = _get_base_policy(mod, block_info)
        if contact_models[mod]["is_recurrent"]:
            policy["policy"] = partial(reduce_recurrent_model, multiplier=multiplier)
        else:
            policy["policy"] = multiplier
        policies[f"{block_info['prefix']}_{mod}"] = policy
    return policies


def reopen_other_models(contact_models, block_info, start_multiplier, end_multiplier):
    """Reduce other contacts with gradually changing multiplier."""
    policies = {}
    other_models = _get_other_models(contact_models)
    for mod in other_models:
        policy = _get_base_policy(mod, block_info)
        policy["policy"] = partial(
            reopen_other_model,
            start_date=block_info["start_date"],
            end_date=block_info["end_date"],
            start_multiplier=start_multiplier,
            end_multiplier=end_multiplier,
            is_recurrent=contact_models[mod]["is_recurrent"],
        )
        policies[f"{block_info['prefix']}_{mod}"] = policy
    return policies


# ======================================================================================
# helpers
# ======================================================================================


def _get_base_policy(affected_model, block_info):
    policy = {
        "affected_contact_model": affected_model,
        "start": block_info["start_date"],
        "end": block_info["end_date"],
    }
    return policy


def _get_educ_models(contact_models):
    return [cm for cm in contact_models if "educ" in cm]


def _get_work_models(contact_models):
    return [cm for cm in contact_models if "work" in cm]


def _get_other_models(contact_models):
    return [cm for cm in contact_models if "other" in cm]
