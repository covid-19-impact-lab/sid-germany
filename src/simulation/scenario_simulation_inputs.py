"""Build simulation inputs that depend on the scenario.

Every public function receives the dependency paths and fixed inputs as arguments
and must return a dictionary with the following entries:

    - contact_policies
    - vaccination_models
    - rapid_test_models
    - rapid_test_reaction_models

"""
import warnings
from functools import partial

import pandas as pd
import yaml

from src.config import BLD
from src.config import SUMMER_SCENARIO_START
from src.config import VERY_LATE
from src.policies.domain_level_policy_blocks import apply_emergency_care_policies
from src.policies.domain_level_policy_blocks import apply_mixed_educ_policies
from src.policies.domain_level_policy_blocks import reduce_educ_models
from src.policies.domain_level_policy_blocks import reduce_work_models
from src.policies.enacted_policies import get_enacted_policies
from src.policies.enacted_policies import get_school_options_for_strict_emergency_care
from src.policies.enacted_policies import HYGIENE_MULTIPLIER
from src.policies.find_people_to_vaccinate import find_people_to_vaccinate
from src.policies.policy_tools import combine_dictionaries
from src.policies.policy_tools import remove_educ_policies
from src.policies.policy_tools import remove_work_policies
from src.policies.policy_tools import shorten_policies
from src.policies.policy_tools import split_policies
from src.testing.rapid_test_reactions import rapid_test_reactions
from src.testing.rapid_tests import rapid_test_demand


def baseline(paths, fixed_inputs):
    baseline_scenario_inputs = {
        "contact_policies": _baseline_policies(fixed_inputs),
        "vaccination_models": _baseline_vaccination_models(paths, fixed_inputs),
        "rapid_test_models": _baseline_rapid_test_models(fixed_inputs),
        "rapid_test_reaction_models": _baseline_rapid_test_reaction_models(
            fixed_inputs
        ),
    }
    return baseline_scenario_inputs


# ================================================================================


def open_all_educ_after_summer_scenario_start(paths, fixed_inputs):
    out = _open_all_educ_after_date(
        paths=paths, fixed_inputs=fixed_inputs, split_date=SUMMER_SCENARIO_START
    )
    return out


def open_all_educ_after_easter(paths, fixed_inputs):
    # day after easter monday as the split date belongs to the 2nd dictionary
    after_easter = "2021-04-06"
    out = _open_all_educ_after_date(
        paths=paths, fixed_inputs=fixed_inputs, split_date=after_easter
    )
    return out


def _open_all_educ_after_date(
    paths, fixed_inputs, split_date, multiplier=HYGIENE_MULTIPLIER
):
    start_date = fixed_inputs["duration"]["start"]
    end_date = fixed_inputs["duration"]["end"]
    contact_models = fixed_inputs["contact_models"]
    enacted_policies = get_enacted_policies(contact_models)

    stays_same, to_change = split_policies(enacted_policies, split_date=split_date)
    after_split_without_educ_policies = remove_educ_policies(to_change)
    block_info = {
        "prefix": f"open_all_educ_after_{pd.Timestamp(split_date).date()}",
        "start_date": split_date,
        "end_date": VERY_LATE,
    }

    new_educ_policies = reduce_educ_models(
        contact_models=contact_models,
        block_info=block_info,
        educ_type="all",
        multiplier=multiplier,
    )
    new_policies = combine_dictionaries(
        [stays_same, after_split_without_educ_policies, new_educ_policies]
    )
    new_policies = shorten_policies(new_policies, start_date, end_date)

    out = {
        "contact_policies": new_policies,
        "vaccination_models": _baseline_vaccination_models(paths, fixed_inputs),
        "rapid_test_models": _baseline_rapid_test_models(fixed_inputs),
        "rapid_test_reaction_models": _baseline_rapid_test_reaction_models(
            fixed_inputs
        ),
    }
    return out


def only_strict_emergency_care_after_april_5(paths, fixed_inputs):
    start_date = fixed_inputs["duration"]["start"]
    end_date = fixed_inputs["duration"]["end"]
    contact_models = fixed_inputs["contact_models"]
    enacted_policies = get_enacted_policies(contact_models)

    # day after easter monday as the split date belongs to the 2nd dictionary
    after_easter = "2021-04-06"
    stays_same, to_change = split_policies(enacted_policies, split_date=after_easter)
    keep = remove_educ_policies(to_change)

    block_info = {
        "prefix": "emergency_care_after_april_5",
        "start_date": after_easter,
        "end_date": VERY_LATE,
    }
    new_young_educ_policies = apply_emergency_care_policies(
        contact_models=contact_models,
        block_info=block_info,
        educ_type="young_educ",
        attend_multiplier=0.25,
        hygiene_multiplier=HYGIENE_MULTIPLIER,
    )
    school_options = get_school_options_for_strict_emergency_care()
    new_school_policies = apply_mixed_educ_policies(
        contact_models=contact_models,
        block_info=block_info,
        educ_type="school",
        **school_options,
    )

    new_policies = combine_dictionaries(
        [stays_same, keep, new_young_educ_policies, new_school_policies]
    )
    new_policies = shorten_policies(new_policies, start_date, end_date)

    out = {
        "vaccination_models": _baseline_vaccination_models(paths, fixed_inputs),
        "rapid_test_models": _baseline_rapid_test_models(fixed_inputs),
        "rapid_test_reaction_models": _baseline_rapid_test_reaction_models(
            fixed_inputs
        ),
        "contact_policies": new_policies,
    }
    return out


# ================================================================================


def no_rapid_tests(paths, fixed_inputs):
    out = {
        "contact_policies": _baseline_policies(fixed_inputs),
        "vaccination_models": _baseline_vaccination_models(paths, fixed_inputs),
        "rapid_test_models": None,
        "rapid_test_reaction_models": None,
    }
    return out


def no_vaccinations_after_feb_15(paths, fixed_inputs):
    start_date = fixed_inputs["duration"]["start"]
    init_start = start_date - pd.Timedelta(31, unit="D")

    vaccination_shares = pd.read_pickle(paths["vaccination_shares"])
    vaccination_models = _get_vaccination_model_with_new_value_after_date(
        vaccination_shares,
        init_start,
        change_date="2021-02-15",
        new_val=5,
        model_name="only_vaccinate_until_feb_15",
    )
    scenario_inputs = {
        "vaccination_models": vaccination_models,
        "contact_policies": _baseline_policies(fixed_inputs),
        "rapid_test_models": _baseline_rapid_test_models(fixed_inputs),
        "rapid_test_reaction_models": _baseline_rapid_test_reaction_models(
            fixed_inputs
        ),
    }
    return scenario_inputs


def vaccinations_after_summer_scenario_start_as_on_strongest_week_day(
    paths, fixed_inputs
):
    """Increase the vaccination rate to that of the best average weekday.

    Averages were taken over the time since family physicians started vaccinating.
    This vaccination rate is extrapolated to every day, including weekends.

    """
    warnings.warn(
        "The ambitious vaccination model abstracts from possible delivery constraints."
    )
    start_date = fixed_inputs["duration"]["start"]
    init_start = start_date - pd.Timedelta(31, unit="D")

    vacc_shares_path = BLD / "data" / "vaccinations" / "mean_vacc_share_per_day.yaml"
    with open(vacc_shares_path) as f:
        vacc_shares = yaml.safe_load(f)

    vaccination_shares = pd.read_pickle(paths["vaccination_shares"])
    vaccination_models = _get_vaccination_model_with_new_value_after_date(
        vaccination_shares,
        init_start,
        change_date=SUMMER_SCENARIO_START,
        new_val=max(vacc_shares.values()),
        model_name="highest_vacc_share_after_summer_scenario_start",
    )
    scenario_inputs = {
        "vaccination_models": vaccination_models,
        "contact_policies": _baseline_policies(fixed_inputs),
        "rapid_test_models": _baseline_rapid_test_models(fixed_inputs),
        "rapid_test_reaction_models": _baseline_rapid_test_reaction_models(
            fixed_inputs
        ),
    }
    return scenario_inputs


def vaccinations_after_easter_as_on_strongest_week_day(paths, fixed_inputs):
    """Increase the vaccination rate to that of the best average weekday.

    Averages were taken over the time since familiy physicians started vaccinating.
    This vaccination rate is extrapolated to every day, including weekends.

    """
    warnings.warn(
        "The ambitious vaccination model abstracts from possible delivery constraints."
    )
    start_date = fixed_inputs["duration"]["start"]
    init_start = start_date - pd.Timedelta(31, unit="D")

    vacc_shares_path = BLD / "data" / "vaccinations" / "mean_vacc_share_per_day.yaml"
    with open(vacc_shares_path) as f:
        vacc_shares = yaml.safe_load(f)

    vaccination_shares = pd.read_pickle(paths["vaccination_shares"])
    vaccination_models = _get_vaccination_model_with_new_value_after_date(
        vaccination_shares,
        init_start,
        change_date="2021-04-06",  # Tuesday after Easter Monday
        new_val=max(vacc_shares.values()),
        model_name="highest_vacc_share_after_easter",
    )
    scenario_inputs = {
        "vaccination_models": vaccination_models,
        "contact_policies": _baseline_policies(fixed_inputs),
        "rapid_test_models": _baseline_rapid_test_models(fixed_inputs),
        "rapid_test_reaction_models": _baseline_rapid_test_reaction_models(
            fixed_inputs
        ),
    }
    return scenario_inputs


def _get_vaccination_model_with_new_value_after_date(
    vaccination_shares, init_start, change_date, new_val, model_name
):
    vaccination_shares[change_date:] = new_val
    vaccination_func = partial(
        find_people_to_vaccinate,
        vaccination_shares=vaccination_shares,
        init_start=init_start,
    )
    vaccination_models = {model_name: {"model": vaccination_func}}
    return vaccination_models


# ================================================================================


def strict_home_office_after_summer_scenario_start(paths, fixed_inputs):
    """Define strict home office after summer scenario starts.
    
    The attendance multiplier is set to 0.54  which corresponds to the mean work
    multiplier in April 2020.
    
    """
    start_date = fixed_inputs["duration"]["start"]
    end_date = fixed_inputs["duration"]["end"]
    contact_models = fixed_inputs["contact_models"]
    enacted_policies = get_enacted_policies(contact_models)

    new_policies = _get_policies_with_different_work_attend_multiplier_after_date(
        enacted_policies=enacted_policies,
        contact_models=contact_models,
        new_attend_multiplier=0.54,
        split_date=SUMMER_SCENARIO_START,
        prefix="work_strict_home_office_after_summer_scenario_start",
    )
    new_policies = shorten_policies(new_policies, start_date, end_date)

    out = {
        "contact_policies": new_policies,
        "vaccination_models": _baseline_vaccination_models(paths, fixed_inputs),
        "rapid_test_models": _baseline_rapid_test_models(fixed_inputs),
        "rapid_test_reaction_models": _baseline_rapid_test_reaction_models(
            fixed_inputs
        ),
    }
    return out


def _get_policies_with_different_work_attend_multiplier_after_date(
    enacted_policies, contact_models, new_attend_multiplier, split_date, prefix
):
    """Set the attend work multiplier to **new_attend_multiplier** after **date**.

    In April 2020, the mean work multiplier was 0.54. In November 2020 it was 0.83.

    """
    stays_same, to_change = split_policies(enacted_policies, split_date=split_date)
    after_split_without_work_policies = remove_work_policies(to_change)

    block_info = {
        "prefix": prefix,
        "start_date": split_date,
        "end_date": VERY_LATE,
    }
    new_work_policies = reduce_work_models(
        contact_models=contact_models,
        block_info=block_info,
        attend_multiplier=new_attend_multiplier,
        hygiene_multiplier=HYGIENE_MULTIPLIER,
    )

    new_policies = combine_dictionaries(
        [stays_same, after_split_without_work_policies, new_work_policies]
    )
    return new_policies


# ================================================================================


def _baseline_policies(fixed_inputs):
    start_date = fixed_inputs["duration"]["start"]
    end_date = fixed_inputs["duration"]["end"]
    policies = get_enacted_policies(fixed_inputs["contact_models"])
    policies = shorten_policies(policies, start_date, end_date)
    return policies


def _baseline_rapid_test_models(fixed_inputs):
    end_date = fixed_inputs["duration"]["end"]
    if end_date <= pd.Timestamp("2021-01-01"):
        rapid_test_models = None
    else:
        rapid_test_models = {
            "standard_rapid_test_demand": {
                "model": rapid_test_demand,
                "start": "2021-01-01",
                "end": VERY_LATE,
            }
        }
    return rapid_test_models


def _baseline_rapid_test_reaction_models(fixed_inputs):
    end_date = fixed_inputs["duration"]["end"]
    if end_date <= pd.Timestamp("2021-01-01"):
        rapid_test_reaction_models = None
    else:
        rapid_test_reaction_models = {
            "rapid_test_reactions": {
                "model": rapid_test_reactions,
                "start": "2021-01-01",
                "end": VERY_LATE,
            }
        }
    return rapid_test_reaction_models


def _baseline_vaccination_models(paths, fixed_inputs):
    start_date = fixed_inputs["duration"]["start"]
    end_date = fixed_inputs["duration"]["end"]
    init_start = start_date - pd.Timedelta(31, unit="D")
    if end_date <= pd.Timestamp("2021-01-01"):
        vaccination_models = None
    else:
        vaccination_shares = pd.read_pickle(paths["vaccination_shares"])
        vaccination_func = partial(
            find_people_to_vaccinate,
            vaccination_shares=vaccination_shares,
            init_start=init_start,
        )
        vaccination_models = {"standard": {"model": vaccination_func}}
    return vaccination_models
