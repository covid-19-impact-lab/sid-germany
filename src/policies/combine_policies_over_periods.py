import pandas as pd

from src.config import BLD
from src.policies.full_policy_blocks import get_lockdown_with_multipliers
from src.policies.policy_tools import combine_dictionaries


def get_jan_to_april_2021_policies(
    contact_models,
    start_date,
    end_date,
    other_multiplier,
    work_multiplier=None,
    work_fill_value=None,
    educ_multiplier=0.8,
):
    """Get policies for January to April 2021.

    Args:
        contat_models (dict)
        start_date: convertible to pandas.Timestamp
        end_date: convertible to pandas.Timestamp
        other_multiplier (float): leisure multiplier to be used for the entire
            time period.
        work_multiplier (float or pandas.Series):
            If None, the google mobility data are used and the user must supply
            a `work_fill_value`.
            If the work_multiplier is a float it's constant for the entire
            time period. If a Series is supplied that Series is used and must
            have as index pandas.date_range(start_date, end_date)

    """
    assert pd.Timestamp(start_date) >= pd.Timestamp(
        "2020-12-27"
    ), "start date must lie after Dec, 26th."
    assert pd.Timestamp(end_date) <= pd.Timestamp(
        "2021-03-31"
    ), "end date must lie before April, 1st."

    work_multiplier = _process_work_multiplier(
        work_multiplier, work_fill_value, start_date, end_date
    )

    to_combine = [
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-02",
                "end_date": "2021-01-31",
                "prefix": "christmas_to_february",
            },
            multipliers={
                "educ": 0.0,
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-02-01",
                "end_date": "2021-02-14",
                "prefix": "first_half_february",
            },
            multipliers={
                "educ": 0.0,
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-02-15",
                "end_date": "2021-02-28",
                "prefix": "2nd_feb_half",
            },
            multipliers={
                "educ": educ_multiplier,
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-03-01",
                "end_date": "2021-03-31",
                "prefix": "march",
            },
            multipliers={
                "educ": educ_multiplier,
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
    ]
    return combine_dictionaries(to_combine)


def get_october_to_christmas_policies(contact_models, educ_multiplier=0.8):
    """Policies from October 1st 2020 until Christmas 2020. """
    work_multiplier_path = BLD / "policies" / "work_multiplier.csv"
    work_multiplier = pd.read_csv(work_multiplier_path, parse_dates=["date"])
    work_multiplier = work_multiplier.set_index("date")["share_working"]

    pre_fall_vacation_multipliers = {
        "educ": educ_multiplier,
        "work": work_multiplier,
        "other": 0.75,
    }
    fall_vacation_multipliers = {
        "educ": educ_multiplier,
        "work": work_multiplier,
        "other": 1.0,
    }
    post_fall_vacation_multipliers = {
        "educ": educ_multiplier,
        "work": work_multiplier,
        "other": 0.65,
    }
    lockdown_light_multipliers = {
        "educ": educ_multiplier,
        "work": work_multiplier,
        "other": 0.45,
    }
    lockdown_light_multipliers_with_fatigue = {
        "educ": educ_multiplier,
        "work": work_multiplier,
        "other": 0.55,
    }
    week_before_christmas_multipliers = {
        "educ": 0.0,
        "work": work_multiplier,
        "other": 0.55,
    }
    to_combine = [
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-01",
                "end_date": "2020-10-09",
                "prefix": "pre_fall_vacation",
            },
            multipliers=pre_fall_vacation_multipliers,
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-10",
                "end_date": "2020-10-23",
                "prefix": "fall_vacation",
            },
            multipliers=fall_vacation_multipliers,
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-24",
                "end_date": "2020-11-01",
                "prefix": "post_fall_vacation",
            },
            multipliers=post_fall_vacation_multipliers,
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-02",
                "end_date": "2020-11-22",
                "prefix": "lockdown_light",
            },
            multipliers=lockdown_light_multipliers,
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-23",
                "end_date": "2020-12-15",
                "prefix": "lockdown_light_with_fatigue",
            },
            multipliers=lockdown_light_multipliers_with_fatigue,
        ),
        # Until start of Christmas vacation
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-16",
                "end_date": "2020-12-20",
                "prefix": "pre-christmas-lockdown-first-half",
            },
            multipliers=week_before_christmas_multipliers,
        ),
        # Until Christmas
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-21",
                "end_date": "2020-12-23",
                "prefix": "pre-christmas-lockdown-second-half",
            },
            multipliers={
                "educ": 0.0,
                "work": work_multiplier,
                "other": 0.35,
            },
        ),
    ]
    return combine_dictionaries(to_combine)


def _process_work_multiplier(work_multiplier, fill_value, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    assert (
        fill_value is None or work_multiplier is None
    ), "fill_value may only be supplied if work_multiplier is None or vice versa"

    if isinstance(work_multiplier, float):
        return pd.Series(data=work_multiplier, index=dates)
    elif isinstance(work_multiplier, pd.Series):
        assert (
            work_multiplier.index == dates
        ).all(), f"Index is not consecutive from {start_date} to {end_date}"
    elif work_multiplier is None:
        default_path = BLD / "policies" / "work_multiplier.csv"
        default = pd.read_csv(default_path, parse_dates=["date"])
        default = default.set_index("date")["share_working"]
        expanded = pd.Series(default, index=dates)
        expanded = expanded.fillna(fill_value)
    return expanded
