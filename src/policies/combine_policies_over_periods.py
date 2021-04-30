import pandas as pd

from src.config import BLD
from src.policies.educ_options_over_time import (
    get_educ_options_1st_half_april,
)
from src.policies.educ_options_over_time import get_educ_options_feb_22_to_march_15
from src.policies.educ_options_over_time import get_educ_options_mid_march_to_easter
from src.policies.educ_options_over_time import (
    graduating_classes_in_a_b_plus_generous_emergency_care,
)
from src.policies.educ_options_over_time import strict_emergency_care
from src.policies.full_policy_blocks import get_lockdown_with_multipliers
from src.policies.policy_tools import combine_dictionaries


def get_enacted_policies_of_2021(
    contact_models,
    scenario_start,
    work_hygiene_multiplier,
    other_multiplier_until_mid_march=0.45,
    other_multiplier_mid_march_until_easter=0.4,
    easter_holiday_other_multiplier=0.25,
):
    """Get enacted policies of 2021.

    This will be updated continuously.

    Args:
        contact_models (dict)
        scenario_start (str): date until which the policies should run.
            Should be of format yyyy-mm-dd.
        other_multiplier_until_mid_march (float): other multiplier until mid of March
        other_multiplier_mid_march_until_easter (float): other multiplier used from
            mid of March until after the Easter holidays.
        easter_holiday_other_multiplier (float): other multiplier used during the easter
            holidays.
        easter_holiday_attend_work_multiplier (float): attend work multiplier used
            during the easter holidays.
        work_hygiene_multiplier (float): work hygiene multiplier used throughout the
            entire period.

    Returns:
        policies (dict)

    """
    last_date = pd.Timestamp("2021-04-06")
    assert pd.Timestamp(scenario_start) <= last_date, (
        "You must update the `get_enacted_policies_of_2021` function to support "
        f"scenario starst after {scenario_start} (only until {last_date.date()}."
    )

    attend_work_multiplier = _get_attend_work_multiplier(scenario_start)
    to_combine = [
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-02",
                "end_date": "2021-01-15",
                "prefix": "january1",
            },
            multipliers={
                "educ": None,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": other_multiplier_until_mid_march,
            },
            **strict_emergency_care(),
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-16",
                "end_date": "2021-02-21",
                "prefix": "january2",
            },
            multipliers={
                "educ": None,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": other_multiplier_until_mid_march,
            },
            **graduating_classes_in_a_b_plus_generous_emergency_care(),
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-02-22",
                "end_date": "2021-03-14",
                "prefix": "educ_reopen_spring_2021",
            },
            multipliers={
                # in fall the educ multiplier was estimated to be 0.8.
                # Here, it is 0.5 to capture additional hygiene measures.
                # This multiplier applies to nurseries and preschools.
                "educ": 0.5,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": other_multiplier_until_mid_march,
            },
            **get_educ_options_feb_22_to_march_15(),
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-03-15",
                "end_date": "2021-04-01",
                "prefix": "mid_march_unitl_easter",
            },
            multipliers={
                "educ": 0.5,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": other_multiplier_mid_march_until_easter,
            },
            **get_educ_options_mid_march_to_easter(),
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                # both dates are inclusive.
                # 2nd April is Good Friday. 5th is Easter Monday
                "start_date": "2021-04-02",
                "end_date": "2021-04-05",
                "prefix": "easter_holidays",
            },
            multipliers={
                "educ": 0.0,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": easter_holiday_other_multiplier,
                **get_educ_options_mid_march_to_easter(),
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-04-06",
                "end_date": "2021-04-18",
                "prefix": "1st_half_april",
            },
            multiplier={
                "educ": 0.5,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                **get_educ_options_1st_half_april(),
            },
        ),
    ]
    return combine_dictionaries(to_combine)


def get_october_to_christmas_policies(
    contact_models,
    educ_options=None,
    educ_multiplier=0.8,
    other_multiplier=None,
    attend_work_multiplier=None,
    work_hygiene_multiplier=1.0,
    work_fill_value=None,
):
    """Policies from October 1st 2020 until Christmas 2020.

    Args:
        educ_options (dict): Nested dictionary with the education types ("school",
            "preschool" or "nursery") that have A/B schooling and/or emergency care as
            keys. Values are dictionaries giving the always_attend_query, a_b_query,
            non_a_b_attend, hygiene_multiplier and a_b_rhythm.
            Note to use the types (e.g. school) and not the contact models
            (e.g. educ_school_1) as keys. The educ_multiplier is not used on top
            of the supplied hygiene multiplier for the contact models covered by the
            educ_options.

            For example:

            .. code-block:: python

                {
                    "school": {
                        "hygiene_multiplier": 0.8,
                        "always_attend_query": "educ_contact_priority > 0.9",
                        "a_b_query": "(age <= 10) | (age >= 16)",
                        "non_a_b_attend": False,
                    }
                }

        educ_multiplier (float): The multiplier for the education contact models
            that are not covered by the a_b_educ_options. This educ_multiplier is
            not used on top of the supplied hygiene multiplier but only used for
            education models that are not in A/B mode.

        work_multiplier (pandas.Series, optional): Series from Oct 1st to Dec 23rd.
            values are between 0 and 1. If not given, the work_multipliers
            implied by the google mobility reports are used.
        work_fill_value (float, optional): If given, the work_multiplier Series will
            be set to this value after November, 1st.

    """
    dates = pd.date_range("2020-10-01", "2020-12-23")
    if attend_work_multiplier is None:
        work_multiplier_path = BLD / "policies" / "work_multiplier.csv"
        attend_work_multiplier = pd.read_csv(work_multiplier_path, parse_dates=["date"])
        attend_work_multiplier = attend_work_multiplier.set_index("date")
        attend_work_multiplier = attend_work_multiplier.loc[dates]
    else:
        assert attend_work_multiplier.between(
            0, 1
        ).all(), "Work multipliers must lie in [0, 1]."
        assert (attend_work_multiplier.index == dates).all(), ""
    if work_fill_value is not None:
        assert 0 <= work_fill_value <= 1, "work fill value must lie in [0, 1]."
        attend_work_multiplier["2020-11-02":] = work_fill_value

    to_combine = [
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-01",
                "end_date": "2020-10-09",
                "prefix": "pre_fall_vacation",
            },
            multipliers={
                "educ": 0.8,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": 0.75,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-10",
                "end_date": "2020-10-23",
                "prefix": "fall_vacation",
            },
            multipliers={
                "educ": 0.8,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": 1.0,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-24",
                "end_date": "2020-11-01",
                "prefix": "post_fall_vacation",
            },
            multipliers={
                "educ": 0.8,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": 0.65,
            },
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-02",
                "end_date": "2020-11-22",
                "prefix": "lockdown_light",
            },
            multipliers={
                "educ": educ_multiplier,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": 0.45 if other_multiplier is None else other_multiplier,
            },
            educ_options=educ_options,
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-23",
                "end_date": "2020-12-15",
                "prefix": "lockdown_light_with_fatigue",
            },
            multipliers={
                "educ": educ_multiplier,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": 0.55 if other_multiplier is None else other_multiplier,
            },
            educ_options=educ_options,
        ),
        # Until start of Christmas vacation
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-16",
                "end_date": "2020-12-20",
                "prefix": "pre-christmas-lockdown-first-half",
            },
            multipliers={
                "educ": None,
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": 0.55 if other_multiplier is None else other_multiplier,
            },
            **strict_emergency_care(),
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
                "work": {
                    "attend_multiplier": attend_work_multiplier,
                    "hygiene_multiplier": work_hygiene_multiplier,
                },
                "other": 0.55 if other_multiplier is None else other_multiplier,
            },
        ),
    ]
    return combine_dictionaries(to_combine)


def _get_attend_work_multiplier(scenario_start):
    work_multiplier_path = BLD / "policies" / "work_multiplier.csv"
    work_multiplier = pd.read_csv(work_multiplier_path, parse_dates=["date"])
    work_multiplier = work_multiplier.set_index("date")
    dates = pd.date_range(pd.Timestamp("2021-01-02"), pd.Timestamp(scenario_start))
    if set(dates).issubset(work_multiplier.index):
        work_multiplier = work_multiplier.loc[dates]
    else:
        work_multiplier = work_multiplier.reindex(dates).fillna(method="ffill")
    return work_multiplier
