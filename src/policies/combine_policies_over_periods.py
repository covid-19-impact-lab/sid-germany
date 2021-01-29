import pandas as pd

from src.config import BLD
from src.policies.full_policy_blocks import get_lockdown_with_multipliers
from src.policies.policy_tools import combine_dictionaries


def get_jan_to_april_2021_policies(
    contact_models, start_date, end_date, other_multiplier, work_multiplier
):
    """Get policies for January to April 2021.

    Args:
        contat_models (dict)
        start_date: convertible to pandas.Timestamp
        end_date: convertible to pandas.Timestamp
        other_multiplier (float): leisure multiplier to be used for the entire
            time period.
        work_multiplier (float or pandas.Series): if it is a float this is the
            value used from the point onward where no estimate is available
            from the google data. If a pandas.Series, missing values of the time
            frame are filled up with google mobility data.


    """
    assert pd.Timestamp(start_date) >= pd.Timestamp(
        "2020-12-27"
    ), "start date must lie after Dec, 26th."
    assert pd.Timestamp(end_date) <= pd.Timestamp(
        "2021-03-31"
    ), "end date must lie before April, 1st."

    work_multiplier = _process_work_multiplier(work_multiplier, start_date, end_date)

    to_combine = [
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-27",
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
                "educ": 0.6,  # old school multiplier
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
                "educ": 0.6,  # old school multiplier
                "work": work_multiplier,
                "other": other_multiplier,
            },
        ),
    ]
    return combine_dictionaries(to_combine)


def get_october_to_christmas_policies(contact_models):
    """Policies from October 1st 2020 until Christmas 2020. """
    work_multiplier_path = BLD / "policies" / "work_multiplier.csv"
    work_multiplier = pd.read_csv(work_multiplier_path, parse_dates=["date"])
    work_multiplier = work_multiplier.set_index("date")["share_working"]

    pre_fall_vacation_multipliers = {
        "educ": 0.8,
        "work": work_multiplier,
        "other": 0.75,
    }
    fall_vacation_multipliers = {
        "educ": 0.8,
        "work": work_multiplier,
        "other": 1.0,
    }
    post_fall_vacation_multipliers = {
        "educ": 0.8,
        "work": work_multiplier,
        "other": 0.65,
    }
    lockdown_light_multipliers = {
        "educ": 0.6,
        "work": work_multiplier,
        "other": 0.45,
    }
    lockdown_light_multipliers_with_fatigue = {
        "educ": 0.6,
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


def _process_work_multiplier(work_multiplier, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    default_path = BLD / "policies" / "work_multiplier.csv"
    default = pd.read_csv(default_path, parse_dates=["date"])
    default = default.set_index("date")["share_working"]
    if isinstance(work_multiplier, pd.Series):
        expanded = pd.Series(work_multiplier, index=dates)
        expanded.fillna(default)
        msg = (
            "NaN remain in your work multipliers after filling them with "
            + "the default work multipliers."
        )
        assert expanded.notnull().all(), msg
    elif isinstance(work_multiplier, float):
        expanded = pd.Series(default, index=dates)
        expanded = expanded.fillna(work_multiplier)
    return expanded
