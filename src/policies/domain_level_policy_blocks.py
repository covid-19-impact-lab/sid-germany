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

from src.policies.single_policy_functions import apply_educ_policy
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
    """Reduce contacts of workers by a multiplier.

    Args:
        multiplier (float or pd.Series): Must be smaller or equal to one. If a
            Series is supplied the index must be dates.

    """
    policies = {}
    work_models = _get_work_models(contact_models)
    for mod in work_models:
        policy = _get_base_policy(mod, block_info)
        policy["policy"] = partial(
            reduce_work_model,
            multiplier=multiplier,
            is_recurrent=contact_models[mod]["is_recurrent"],
        )
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
            is_recurrent=contact_models[mod]["is_recurrent"],
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
        policy["policy"] = partial(
            shut_down_model, is_recurrent=contact_models[mod]["is_recurrent"]
        )
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
            policy["policy"] = multiplier

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


def implement_general_schooling_policy(
    contact_models,
    block_info,
    educ_options,
    other_educ_multiplier,
):
    """Split education groups for some children and apply a hygiene multiplier.

    Args:
        educ_options (dict): Nested dictionary with the education types ("school",
            "preschool" or "nursery") that have A/B schooling and/or emergency care as
            keys. Values are dictionaries giving the always_attend_query, a_b_query,
            non_a_b_attend, hygiene_multiplier and a_b_rhythm.
            Note to use the types (e.g. school) and not the contact models
            (e.g. educ_school_1) as keys. The other_educ_multiplier is not used on top
            of the supplied hygiene multiplier for the contact models covered by the
            educ_options.
            For example:
            {
                "school": {
                    "hygiene_multiplier": 0.8,
                    "always_attend_query": "educ_contact_priority > 0.9",
                    "a_b_query": "(age <= 10) | (age >= 16)",
                    "non_a_b_attend": False,
            }

        other_educ_multiplier (float): multiplier for the contact models that have
            neither A/B schooling nor emergency care, i.e. which do not have an entry
            in educ_options.


    """
    policies = {}
    educ_models = _get_educ_models(contact_models)
    for mod in educ_models:
        policy = _get_base_policy(mod, block_info)
        educ_type = _determine_educ_type(mod)
        if educ_type in educ_options:
            assert contact_models[mod]["is_recurrent"], (
                "apply_educ_policy only available for recurrent models, "
                f"{mod} is non-recurrent."
            )
            policy["policy"] = partial(
                apply_educ_policy,
                group_id_column=contact_models[mod]["assort_by"][0],
                **educ_options[educ_type],
            )
        else:
            assert (
                0 <= other_educ_multiplier <= 1
            ), "Only multipliers in [0, 1] allowed."
            if other_educ_multiplier == 0:
                policy["policy"] = partial(
                    shut_down_model, is_recurrent=contact_models[mod]["is_recurrent"]
                )
            # currently all educ models are recurrent but don't want to assume it
            elif contact_models[mod]["is_recurrent"]:
                policy["policy"] = partial(
                    reduce_recurrent_model, multiplier=other_educ_multiplier
                )
            else:
                policy["policy"] = other_educ_multiplier
        policies[f"{block_info['prefix']}_{mod}"] = policy
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
        policy["policy"] = partial(
            shut_down_model, is_recurrent=contact_models[mod]["is_recurrent"]
        )
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


def _determine_educ_type(model_name):
    """Determine whether an education model is school, preschool or nursery.

    Args:
        model_name (str): name of the education model, e.g. educ_school_0.

    """
    name_parts = model_name.split("_")
    msg = f"The name of your education model {model_name} does not "
    assert len(name_parts) == 3, msg + "consist of three parts."
    assert name_parts[0] == "educ", (
        msg + f"have educ as first part but {name_parts[0]}."
    )
    assert name_parts[1] in ["nursery", "preschool", "school"], (
        msg + "belong to ['nursery', 'preschool', 'school']."
    )
    assert name_parts[2].isdigit(), (
        msg + f"have a number as last part but {name_parts[2]}"
    )
    return name_parts[1]
