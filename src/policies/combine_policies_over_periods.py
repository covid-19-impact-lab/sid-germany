from src.policies.full_policy_blocks import get_soft_lockdown
from src.policies.policy_tools import combine_dictionaries


def get_october_to_christmas_policies(contact_models):
    """Policies from October 1st 2020 until Christmas 2020. """
    pre_fall_vacation_multipliers = {"educ": 0.8, "work": 0.775, "other": 0.75}
    fall_vacation_multipliers = {"educ": 0.8, "work": 0.63, "other": 1.0}
    post_fall_vacation_multipliers = {"educ": 0.8, "work": 0.775, "other": 0.65}
    lockdown_light_multipliers = {"educ": 0.6, "work": 0.73 * 0.95, "other": 0.45}
    lockdown_light_multipliers_with_fatigue = {
        "educ": 0.6,
        "work": 0.76 * 0.95,
        "other": 0.55,
    }
    week_before_christmas_multipliers = {
        "educ": 0.0,
        "work": 0.76 * 0.95,
        "other": 0.55,
    }
    to_combine = [
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-01",
                "end_date": "2020-10-09",
                "prefix": "pre_fall_vacation",
            },
            multipliers=pre_fall_vacation_multipliers,
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-10",
                "end_date": "2020-10-23",
                "prefix": "fall_vacation",
            },
            multipliers=fall_vacation_multipliers,
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-24",
                "end_date": "2020-11-01",
                "prefix": "post_fall_vacation",
            },
            multipliers=post_fall_vacation_multipliers,
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-02",
                "end_date": "2020-11-22",
                "prefix": "lockdown_light",
            },
            multipliers=lockdown_light_multipliers,
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-23",
                "end_date": "2020-12-15",
                "prefix": "lockdown_light_with_fatigue",
            },
            multipliers=lockdown_light_multipliers_with_fatigue,
        ),
        # Until start of christmas vacation
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-16",
                "end_date": "2020-12-20",
                "prefix": "pre-christmas-lockdown-first-half",
            },
            multipliers=week_before_christmas_multipliers,
        ),
        # Until Christmas
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-21",
                "end_date": "2020-12-23",
                "prefix": "pre-christmas-lockdown-second-half",
            },
            multipliers={
                "educ": 0.0,
                "work": 0.15,
                "other": 0.35,
            },
        ),
    ]
    return combine_dictionaries(to_combine)
