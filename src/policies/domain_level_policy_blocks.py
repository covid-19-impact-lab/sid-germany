"""Functions to create building blocks of policies with common dates and domains.

Currently we distinguish the domains "work", "educ" and "other". "households" are not
included in any domain because they are currently not subject to policies.

The functions here should not be called directly but will be used in the module
``full_policy_blocks.py``.

All public functions here take the following arguments (and possibly some more)

- contact_models (:obj:`dict`): A sid compatible dictionary with contact models.
- block_info (:obj:`dict`): A dictionary containing start_date, end_date and prefix of a
  block of policies.

The functions here expect that the domain names are part of contact model names.

"""
from functools import partial

from src.policies.single_policy_functions import mixed_educ_policy
from src.policies.single_policy_functions import reduce_recurrent_model
from src.policies.single_policy_functions import reduce_work_model
from src.policies.single_policy_functions import reopen_other_model
from src.policies.single_policy_functions import shut_down_model

# ======================================================================================
# work policies
# ======================================================================================


def reduce_work_models(
    contact_models, block_info, attend_multiplier, hygiene_multiplier
):
    """Reduce contacts of workers by a attend_multiplier.

    Args:
        attend_multiplier (float or pd.Series): Must be smaller or equal to one. If a
            Series is supplied the index must be dates.
        hygiene_multiplier (float or pd.Series): Must be smaller or equal to one. If a
            Series is supplied the index must be dates.

    """
    policies = {}
    work_models = _get_work_models(contact_models)
    for mod in work_models:
        policy = _get_base_policy(mod, block_info)
        policy["policy"] = partial(
            reduce_work_model,
            attend_multiplier=attend_multiplier,
            hygiene_multiplier=hygiene_multiplier,
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


def reduce_educ_models(contact_models, block_info, educ_type, multiplier):
    """Apply a simple hygiene multiplier to a subset of the educ models.

    Args:
        contact_models (dict): sid contact models
        block_info (dict): keys are 'start_date', 'end_date' and 'prefix'.
        educ_type (str): "young_educ" or "school". The function
            f"_get_{educ_type}_models" must exist in the local namespace.
        multiplier (float): value of the hygiene multiplier.

    """
    policies = {}
    # use globals because the functions are in the global namespace and not imported
    get_relevant_models = globals()[f"_get_{educ_type}_models"]
    relevant_models = get_relevant_models(contact_models)
    for model in relevant_models:
        policy = _get_base_policy(model, block_info)
        if contact_models[model]["is_recurrent"]:
            policy["policy"] = partial(reduce_recurrent_model, multiplier=multiplier)
        else:
            policy["policy"] = multiplier
        policies[f"{block_info['prefix']}_{model}"] = policy

    return policies


def apply_mixed_educ_policies(contact_models, block_info, educ_type, educ_options):
    """Apply mixed_educ_policies to a set of contact models.

    Args:
        contact_models (dict): sid contact models
        block_info (dict): keys are 'start_date', 'end_date' and 'prefix'.
        educ_type (str): "young_educ" or "school". The function
            f"_get_{educ_type}_models" must exist in the local namespace.
        educ_options (dict): keys must contain 'always_attend_query', 'a_b_query',
            'non_a_b_attend' and 'hygiene_multiplier'. 'a_b_rhythm' is an
            optional key. See `mixed_educ_policy` for details.

    """
    policies = {}
    # use globals because the functions are in the global namespace and not imported
    get_relevant_models = globals()[f"_get_{educ_type}_models"]
    relevant_models = get_relevant_models(contact_models)
    for model in relevant_models:
        policy = _get_base_policy(model, block_info)
        if contact_models[model]["is_recurrent"]:
            policy["policy"] = partial(
                mixed_educ_policy,
                group_id_column=contact_models[model]["assort_by"][0],
                **educ_options,
            )
        else:
            raise ValueError(
                "mixed_educ_policies can only be applied to recurrent models. "
                f"{model} is not recurrent."
            )
        policies[f"{block_info['prefix']}_{model}"] = policy
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


def _get_young_educ_models(contact_models):
    return [cm for cm in contact_models if "nursery" in cm or "preschool" in cm]


def _get_school_models(contact_models):
    return [cm for cm in contact_models if "school" in cm and "preschool" not in cm]


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
