import pandas as pd

from src.policies.full_policy_blocks import get_soft_lockdown
from src.policies.policy_tools import combine_dictionaries


def get_jan_to_april_2021_policies(
    contact_models,
    start_date,
    end_date,
    other_multiplier,
):
    "Get policies for January and February 2021."
    assert pd.Timestamp(start_date) >= pd.Timestamp(
        "2020-12-27"
    ), "start date must lie after Dec, 26th."
    assert pd.Timestamp(end_date) <= pd.Timestamp(
        "2021-03-31"
    ), "end date must lie after before April, 1st."

    hygiene_multiplier = 0.95  # compared to October
    work_multiplier = 0.33 + 0.66 * 0.55 * hygiene_multiplier

    to_combine = [
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-27",
                "end_date": "2021-01-03",
                "prefix": "post-christmas-lockdown",
            },
            multipliers={
                "educ": 0.0,
                # not in line with google mobility data!
                "work": 0.33 + 0.66 * 0.15,
                "other": other_multiplier,
            },
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-04",
                "end_date": "2021-01-11",
                "prefix": "after-christmas-vacation",
            },
            multipliers={
                "educ": 0.0,
                # google mobility data says work mobility -40% !!!
                "work": 0.33 + 0.66 * 0.4 * hygiene_multiplier,
                "other": other_multiplier,
            },
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-12",
                "end_date": "2021-02-14",
                "prefix": "mid_jan_to_mid_feb",
            },
            multipliers={
                "educ": 0.0,
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-02-15",
                "end_date": "2021-02-28",
                "prefix": "2nd_feb_half",
            },
            multipliers={
                "educ": 0.6,  # old school multiplier
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-03-01",
                "end_date": "2021-03-15",
                "prefix": "1st_half_march",
            },
            multipliers={
                "educ": 0.6,  # old school multiplier
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-03-16",
                "end_date": "2021-03-31",
                "prefix": "2nd_half_march",
            },
            multipliers={
                "educ": 0.6,  # old school multiplier
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
    ]
    return combine_dictionaries(to_combine)


def get_october_to_christmas_policies(contact_models):
    """Policies from October 1st 2020 until Christmas 2020. """
    hygiene_multiplier = 0.95  # compared to October
    pre_fall_vacation_multipliers = {
        "educ": 0.8,
        "work": 0.33 + 0.66 * 0.775,
        "other": 0.75,
    }
    fall_vacation_multipliers = {
        "educ": 0.8,
        "work": 0.33 + 0.66 * 0.63,
        "other": 1.0,
    }
    post_fall_vacation_multipliers = {
        "educ": 0.8,
        "work": 0.33 + 0.66 * 0.775,
        "other": 0.65,
    }
    lockdown_light_multipliers = {
        "educ": 0.6,
        "work": 0.33 + 0.66 * 0.73 * hygiene_multiplier,
        "other": 0.45,
    }
    lockdown_light_multipliers_with_fatigue = {
        "educ": 0.6,
        "work": 0.33 + 0.66 * 0.76 * hygiene_multiplier,
        "other": 0.55,
    }
    week_before_christmas_multipliers = {
        "educ": 0.0,
        "work": 0.33 + 0.66 * 0.76 * hygiene_multiplier,
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
                # from google mobility data
                "work": 0.5,
                "other": 0.35,
            },
        ),
    ]
    return combine_dictionaries(to_combine)
