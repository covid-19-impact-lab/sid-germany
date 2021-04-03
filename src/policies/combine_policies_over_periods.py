import pandas as pd

from src.config import BLD
from src.policies.full_policy_blocks import get_lockdown_with_multipliers
from src.policies.policy_tools import combine_dictionaries


def get_educ_options_mid_march_to_easter(hygiene_multiplier=0.5):
    """Get the educ_options starting March 15th.

    Situation:
        - BY:
            - source: https://bit.ly/3lOZowy
            - <50 incidence: normal schooling
            - 50-100 incidence: A/B schooling
            - >100 incidence: distance for all except graduation classes

        - BW:
            - source: https://km-bw.de/Coronavirus (accessed: March 25th)
            - primaries and 5th, 6th grade open normally since 15/3
            - graduating classes open since 22/2
            - rest continues distance learning

        - NRW:
            - source: https://bit.ly/3f9O4Kp (WDR)
            - A/B schooling since March 15th

    -> We simplify this to A/B schooling for everyone plus emergency care
       and graduating classes

    """
    primary_emergency_query = "(educ_contact_priority > 0.66 & age < 10)"
    secondary_emergency_query = "(educ_contact_priority > 0.75 & age >= 10)"
    always_attend_query = f"{primary_emergency_query} | {secondary_emergency_query}"
    educ_options = {
        "school": {
            "hygiene_multiplier": hygiene_multiplier,
            # Demand seems to be lower the older the children
            # but only data from Bavaria available: https://bit.ly/3sGHZbJ
            "always_attend_query": always_attend_query,
            # simplify to A/B schooling for everyone
            "a_b_query": "age == age",
            "non_a_b_attend": False,
            # only very anecdotally the current most common a_b_rhythm.
            "a_b_rhythm": "daily",
        }
    }
    return {"educ_options": educ_options}


def _get_educ_options_feb_22_to_march_15(hygiene_multiplier=0.5):
    """Get the educ_options from February 22nd to March 15th.

    This assumes that nurseries and preschools are open normally (i.e. only the
    general educ_multiplier is applied to them). Schools open for primary students
    and graduating classes in A/B while maintaining emergency care for children
    with high educ_contact_priority (>0.9 for secondary students <13 and >0.66 for
    primary students).

    Summary of actual policies of states with >=8 mio inhabitants:
        BW: nurseries and primary schools open Feb 22nd. No mention of preschools.
            graduating classes attend.
        BY: nurseries and primary schools open Feb 22nd. No mention of preschools.
            graduating classes attend.
        NRW: nurseries pretty normal since Feb 22nd. Primaries and graduating classes
            start Feb 22nd, A/B or full depending on local incidecne.
            (https://bit.ly/3uSp6Ey)

    Sources:
        - https://www.mdr.de/brisant/corona-schule-geoeffnet-100.html
        - https://bit.ly/2O3aS3h

    Args:
        hygiene_multiplier (float): Hygiene multiplier for school children that
            attend because they have a right to emergency care.

    """
    primary_emergency_query = "(educ_contact_priority > 0.66 & age < 10)"
    secondary_emergency_query = "(educ_contact_priority > 0.75 & age >= 10)"
    always_attend_query = f"{primary_emergency_query} | {secondary_emergency_query}"
    educ_options = {
        "school": {
            "hygiene_multiplier": hygiene_multiplier,
            # Demand seems to be lower the older the children
            # but only data from Bavaria available: https://bit.ly/3sGHZbJ
            "always_attend_query": always_attend_query,
            # primary schools and graduating classes in A/B mode
            "a_b_query": "age <= 10 | age >= 16",
            "non_a_b_attend": False,
            # only very anecdotally the current most common a_b_rhythm.
            "a_b_rhythm": "daily",
        }
    }
    return {"educ_options": educ_options}


def _graduating_classes_in_a_b_plus_generous_emergency_care(
    young_children_multiplier=0.8, school_hygiene_multiplier=0.5
):
    """Get expanded emergency care with graduating classes in A/B schooling.

    This is what was in effect in the 2nd half of January.

    In the second half of January, approx. 1 / 3 of children below secondary
    level were in emergency care

    sources:
        - https://bit.ly/3uGL1Pb
        - https://bit.ly/2PErr5T
        - Berlin: <40% (https://bit.ly/304R5ml)
        - Niedersachsen: 38% (https://bit.ly/2PtPdSb)

    In addition, many states opened graduating classes. We open them at the federal
    level in A / B schooling on Jan 16th. This abstracts from state variety:
        - Bavaria: started on 1 Feb (https://bit.ly/3e4p1YE)
        - Baden-Württemberg started on 18 Jan (https://bit.ly/2Ofq9O7)
        - Berlin started on  11 Jan (https://bit.ly/385iCZk)

    sources:
        - https://taz.de/Schulen-in-Coronazeiten/!5753515/
        - https://tinyurl.com/2jfm4tp8

    Args:
        young_children_multiplier (float): hygiene multiplier for children in
            emergency care in preschools and nurseries. Higher by default than
            in primaries and secondary schools because usually there are less
            mask requirements for younger children and masks don't fit them as well.
        school_hygiene_multiplier (float): hygiene multiplier for children in
            emergency care in schools and graduating classes that attend in an
            A/B schooling mode.

    """
    primary_emergency_query = "(educ_contact_priority > 0.66 & age < 10)"
    secondary_emergency_query = "(educ_contact_priority > 0.9 & age >= 10)"
    always_attend_query = f"{primary_emergency_query} | {secondary_emergency_query}"

    educ_options = {
        "school": {
            "hygiene_multiplier": school_hygiene_multiplier,
            # Demand seems to be lower the older the children
            # but only data from Bavaria available: https://bit.ly/3sGHZbJ
            "always_attend_query": always_attend_query,
            "non_a_b_attend": False,
            # to cover graduating classes
            "a_b_query": "age in [16, 17, 18]",
            # only very anecdotally the current most common a_b_rhythm.
            "a_b_rhythm": "daily",
        },
        "preschool": {
            "hygiene_multiplier": young_children_multiplier,
            "always_attend_query": "educ_contact_priority > 0.67",
            "non_a_b_attend": False,
            "a_b_query": False,
        },
        "nursery": {
            "hygiene_multiplier": young_children_multiplier,
            "always_attend_query": "educ_contact_priority > 0.67",
            "non_a_b_attend": False,
            "a_b_query": False,
        },
    }
    return {"educ_options": educ_options}


def strict_emergency_care(hygiene_multiplier=0.8):
    """Get educ options with limited emergency care (as for example around vacations).

    This is based on the data for the 1st half of January where many parents might
    have still had vacations.
    1 in 10 primary children attend in emergency care (source only
    for Bavaria, mid January: https://bit.ly/3sGHZbJ)

    Jump from ~25% to 33% between first Jan week and later (https://bit.ly/3uGL1Pb).
    This could be a vacation effect with more parents returning to work.

    For the first half of Jaunary this abstracts from graduating classes opening
    early in some states, such as Berlin (11 of Jan, https://bit.ly/385iCZk).

    """
    primary_emergency_query = "(educ_contact_priority > 0.75 & age < 10)"
    secondary_emergency_query = "(educ_contact_priority > 0.95 & age >= 10)"
    always_attend_query = f"{primary_emergency_query} | {secondary_emergency_query}"

    educ_options = {
        "school": {
            "hygiene_multiplier": hygiene_multiplier,
            "always_attend_query": always_attend_query,
            "a_b_query": False,
            "non_a_b_attend": False,
        },
        "preschool": {
            "hygiene_multiplier": hygiene_multiplier,
            "always_attend_query": "educ_contact_priority > 0.75",
            "a_b_query": False,
            "non_a_b_attend": False,
        },
        "nursery": {
            "hygiene_multiplier": hygiene_multiplier,
            "always_attend_query": "educ_contact_priority > 0.75",
            "a_b_query": False,
            "non_a_b_attend": False,
        },
    }
    return {"educ_options": educ_options}


def get_enacted_policies_of_2021(
    contact_models,
    scenario_start,
    other_multiplier_until_mid_march=0.45,
    other_multiplier_mid_march_until_easter=0.4,
    easter_holiday_other_multiplier=0.15,
    easter_holiday_work_multiplier=0.15,
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
        easter_holiday_work_multiplier (float): work multiplier used during the easter
            holidays.

    Returns:
        policies (dict)

    """
    last_date = pd.Timestamp("2021-04-06")
    assert pd.Timestamp(scenario_start) <= last_date, (
        "You must update the `get_enacted_policies_of_2021` function to support "
        f"scenario starst after {scenario_start} (only until {last_date.date()}."
    )

    work_multiplier = _get_work_multiplier(scenario_start)
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
                "work": work_multiplier,
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
                "work": work_multiplier,
                "other": other_multiplier_until_mid_march,
            },
            **_graduating_classes_in_a_b_plus_generous_emergency_care(),
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
                "work": work_multiplier,
                "other": other_multiplier_until_mid_march,
            },
            **_get_educ_options_feb_22_to_march_15(),
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
                "work": work_multiplier,
                "other": other_multiplier_mid_march_until_easter,
            },
            **get_educ_options_mid_march_to_easter(),
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                # both dates are inclusive.
                # 2nd April is Good Friday. 4th is Easter Monday
                "start_date": "2021-04-02",
                "end_date": "2021-04-04",
                "prefix": "easter_holidays",
            },
            multipliers={
                "educ": 0.0,
                "work": easter_holiday_work_multiplier,
                "other": easter_holiday_other_multiplier,
                **get_educ_options_mid_march_to_easter(),
            },
        ),
    ]
    return combine_dictionaries(to_combine)


def get_october_to_christmas_policies(
    contact_models,
    educ_options=None,
    educ_multiplier=0.8,
    other_multiplier=None,
    work_multiplier=None,
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
            {
                "school": {
                    "hygiene_multiplier": 0.8,
                    "always_attend_query": "educ_contact_priority > 0.9",
                    "a_b_query": "(age <= 10) | (age >= 16)",
                    "non_a_b_attend": False,
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
    if work_multiplier is None:
        work_multiplier_path = BLD / "policies" / "work_multiplier.csv"
        work_multiplier = pd.read_csv(work_multiplier_path, parse_dates=["date"])
        work_multiplier = work_multiplier.set_index("date")
        work_multiplier = work_multiplier.loc[dates]
    else:
        assert work_multiplier.between(
            0, 1
        ).all(), "Work multipliers must lie in [0, 1]."
        assert (work_multiplier.index == dates).all(), ""
    if work_fill_value is not None:
        assert 0 <= work_fill_value <= 1, "work fill value must lie in [0, 1]."
        work_multiplier["2020-11-02":] = work_fill_value

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
                "work": work_multiplier,
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
                "work": work_multiplier,
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
                "work": work_multiplier,
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
                "work": work_multiplier,
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
                "work": work_multiplier,
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
                "work": work_multiplier,
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
                "work": work_multiplier,
                "other": 0.55 if other_multiplier is None else other_multiplier,
            },
        ),
    ]
    return combine_dictionaries(to_combine)


def _get_work_multiplier(scenario_start):
    work_multiplier_path = BLD / "policies" / "work_multiplier.csv"
    work_multiplier = pd.read_csv(work_multiplier_path, parse_dates=["date"])
    work_multiplier = work_multiplier.set_index("date")
    dates = pd.date_range(pd.Timestamp("2021-01-02"), pd.Timestamp(scenario_start))
    if set(dates).issubset(work_multiplier.index):
        work_multiplier = work_multiplier.loc[dates]
    else:
        work_multiplier = work_multiplier.reindex(dates).fillna(method="ffill")
    return work_multiplier
