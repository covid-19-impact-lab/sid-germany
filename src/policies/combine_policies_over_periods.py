import pandas as pd

from src.config import BLD
from src.policies.full_policy_blocks import get_lockdown_with_multipliers
from src.policies.policy_tools import combine_dictionaries


def _get_jan_first_half_educ_options(hygiene_multiplier=0.8):
    """Get the a_b_educ_options and emergency_options for the 1st half of January.

    1 in 10 children in 5th and 6th grade attends in emergency care (source only
    for Bavaria, mid January: https://bit.ly/3sGHZbJ)

    Jump from ~25% to 33% between first Jan week and later (https://bit.ly/3uGL1Pb).
    This could be a vacation effect with more parents returning to work.

    This abstracts from graduating classes opening early in some states, such as
    Berlin (11 of Jan, https://bit.ly/385iCZk).

    """
    emergency_options = {
        "school": {
            "hygiene_multiplier": hygiene_multiplier,
            # emergency care only until 6th grade
            "always_attend_query": "educ_contact_priority > 0.9",
        },
        "preschool": {
            "hygiene_multiplier": hygiene_multiplier,
            "always_attend_query": "educ_contact_priority > 0.75",
        },
        "nursery": {
            "hygiene_multiplier": hygiene_multiplier,
            "always_attend_query": "educ_contact_priority > 0.75",
        },
    }
    educ_options = {"a_b_educ_options": None, "emergency_options": emergency_options}
    return educ_options


def _get_mid_jan_to_mid_feb_educ_options(
    young_children_multiplier=0.8, school_multiplier=0.5
):
    """Get the a_b_educ_options and emergency_options for the 2nd half of January.

    In the secnd half of January, approx. 1 / 3 of children were in emergency care
    (https://bit.ly/3uGL1Pb).

    sources:
        - Berlin: <40% (https://bit.ly/304R5ml)
        - Niedersachsen: 38% (https://bit.ly/2PtPdSb)

    In addition, many states opened graduating classes. We open them at the federal
    level in A / B schooling on Jan 16th. This abstracts from state variety:
        - Bavaria: started on 1 Feb (https://bit.ly/3e4p1YE)
        - Baden-Württemberg started on 18 Jan (https://bit.ly/2Ofq9O7)
        - Berlin started on  11 Jan (https://bit.ly/385iCZk)

    Sources:
        - https://taz.de/Schulen-in-Coronazeiten/!5753515/
        - https://tinyurl.com/2jfm4tp8

    """
    emergency_options = {
        "preschool": {
            "hygiene_multiplier": young_children_multiplier,
            "always_attend_query": "educ_contact_priority > 0.67",
        },
        "nursery": {
            "hygiene_multiplier": young_children_multiplier,
            "always_attend_query": "educ_contact_priority > 0.67",
        },
    }
    a_b_educ_options = {
        "school": {
            "others_attend": False,
            "hygiene_multiplier": school_multiplier,
            # to cover graduating classes
            "subgroup_query": "age in [16, 17, 18]",
            # Demand seems to be lower the older the children
            # but only data from Bavaria available: https://bit.ly/3sGHZbJ
            "always_attend_query": "educ_contact_priority > 0.9",
            # only very anecdotally the current most common rhythm.
            "rhythm": "daily",
        }
    }
    educ_options = {
        "a_b_educ_options": a_b_educ_options,
        "emergency_options": emergency_options,
    }
    return educ_options


def _get_educ_options_starting_feb_22(school_multiplier=0.5):
    """Get the a_b_educ_options and emergency_options from 22 Feb onwards.

    This assumes that nurseries and preschools are open normally (i.e. only the
    general educ_multiplier is applied to them). Schools open for primary students
    and graduating classes in A/B while maintaining emergency care for children
    with very high educ_contact_priority (>0.9).

    Summary of actual policies of states with >=8 mio inhabitants:
        BW: nurseries and primary schools open Feb 22nd. No mention of preschools.
            graduating classes attend.
        BY: nurseries and primary schools open Feb 22nd. No mention of preschools.
            graduating classes attend.
        Niedersachsen: primaries and graduating classes in A/B since January.
            nurseries only in emergency care. No mention of preschools.
        NRW: nurseries pretty normal since Feb 22nd. Primaries and graduating classes
            start Feb 22nd, A/B or full depending on local incidecne.
            (https://bit.ly/3uSp6Ey)

    Sources:
        - https://www.mdr.de/brisant/corona-schule-geoeffnet-100.html
        - https://bit.ly/2O3aS3h

    """
    emergency_options = {}
    a_b_educ_options = {
        "school": {
            "others_attend": False,
            "hygiene_multiplier": school_multiplier,
            # to cover graduating classes
            "subgroup_query": "age in [16, 17, 18]",
            # Demand seems to be lower the older the children
            # but only data from Bavaria available: https://bit.ly/3sGHZbJ
            "always_attend_query": "educ_contact_priority > 0.9",
            # only very anecdotally the current most common rhythm.
            "rhythm": "daily",
        }
    }
    educ_options = {
        "a_b_educ_options": a_b_educ_options,
        "emergency_options": emergency_options,
    }
    return educ_options


def get_enacted_policies_of_2021(
    contact_models,
    scenario_start,
    other_multiplier=0.45,
):
    """Get enacted policies of 2021.

    This will be updated continuously.

    Args:
        contact_models (dict)
        scenario_start (str): date until which the policies should run.
            Should be of format yyyy-mm-dd.

    Returns:
        policies (dict)

    """
    assert pd.Timestamp(scenario_start) <= pd.Timestamp("2021-03-01"), (
        "You must update the `get_enacted_policies_of_2021` function to support "
        "such a late scenario start."
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
                "other": other_multiplier,
            },
            **_get_jan_first_half_educ_options(),
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
                "other": other_multiplier,
            },
            **_get_mid_jan_to_mid_feb_educ_options(),
        ),
        get_lockdown_with_multipliers(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-02-22",
                "end_date": scenario_start,
                "prefix": "educ_reopen_spring_2021",
            },
            multipliers={
                # in fall the educ multiplier was estimated to be 0.8.
                # Here, it is 0.5 to capture additional hygiene measures.
                # This multiplier applies to nurseries and preschools.
                "educ": 0.5,
                "work": work_multiplier,
                "other": other_multiplier,
            },
            **_get_educ_options_starting_feb_22(),
        ),
    ]
    return combine_dictionaries(to_combine)


def get_october_to_christmas_policies(
    contact_models,
    a_b_educ_options=None,
    educ_multiplier=0.8,
    other_multiplier=None,
    work_multiplier=None,
    work_fill_value=None,
):
    """Policies from October 1st 2020 until Christmas 2020.

    Args:
        a_b_educ_options (dict): For every education type ("school", "preschool",
            "nursery") that is in an A/B schooling mode, add name of the type
            as key and the others_attend, hygiene_multiplier and - if desired -
            the subgroup_query, always_attend_query and rhythm as key-value dict.
            Note to use the types (e.g. school) and not the contact models
            (e.g. educ_school_1) as keys. The educ_multiplier is not used on top
            of the supplied hygiene multiplier but only used for education models
            that are not in A/B mode. Default is no A/B education.
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
    if a_b_educ_options is None:
        a_b_educ_options = {}
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
            a_b_educ_options=a_b_educ_options,
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
            a_b_educ_options=a_b_educ_options,
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
                "educ": 0.0,
                "work": work_multiplier,
                "other": 0.55 if other_multiplier is None else other_multiplier,
            },
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
