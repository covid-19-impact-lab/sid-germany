import pandas as pd

from src.config import BLD
from src.policies.domain_level_policy_blocks import apply_emergency_care_policies
from src.policies.domain_level_policy_blocks import apply_mixed_educ_policies
from src.policies.domain_level_policy_blocks import reduce_educ_models
from src.policies.domain_level_policy_blocks import reduce_other_models
from src.policies.domain_level_policy_blocks import reduce_work_models
from src.policies.policy_tools import combine_dictionaries

VERY_EARLY = pd.Timestamp("2020-01-01")
VERY_LATE = pd.Timestamp("2022-12-31")
HYGIENE_MULTIPLIER = 0.7
"""Hygiene multiplier for educ and work models. Is in effect from November 2020 on."""


def get_enacted_policies(contact_models):
    """Get enacted policies."""
    work_policies = _get_enacted_work_policies(contact_models)
    other_policies = _get_enacted_other_policies(contact_models)
    young_educ_policies = _get_enacted_young_educ_policies(contact_models)
    school_policies = _get_enacted_school_policies(contact_models)
    policies = combine_dictionaries(
        [work_policies, school_policies, young_educ_policies, other_policies]
    )
    return policies


def _get_enacted_work_policies(contact_models):
    """Get the enacted work policies.

    This uses the google mobility data as proxy for the share of individuals
    working in home office. The last value is extrapolated into the future.

    """
    multiplier_path = BLD / "policies" / "work_multiplier.csv"
    attend_multiplier = pd.read_csv(multiplier_path, parse_dates=["date"])
    attend_multiplier = attend_multiplier.set_index("date")
    dates = pd.date_range(VERY_EARLY, VERY_LATE)
    attend_multiplier = attend_multiplier.reindex(dates)
    attend_multiplier = attend_multiplier.fillna(method="ffill").fillna(method="bfill")

    before_november_policies = reduce_work_models(
        contact_models=contact_models,
        block_info={
            "start_date": VERY_EARLY,
            "end_date": "2020-11-01",
            "prefix": "before_november_2020",
        },
        attend_multiplier=attend_multiplier,
        hygiene_multiplier=1.0,
    )

    after_november_policies = reduce_work_models(
        contact_models=contact_models,
        block_info={
            "start_date": "2020-11-02",
            "end_date": VERY_LATE,
            "prefix": "after_november_2020",
        },
        attend_multiplier=attend_multiplier,
        hygiene_multiplier=HYGIENE_MULTIPLIER,
    )
    work_policies = combine_dictionaries(
        [before_november_policies, after_november_policies]
    )
    return work_policies


def _get_enacted_other_policies(contact_models):
    """Get enacted other policies.

    These multipliers are on top of the implemented seasonality.

    """
    specs = [
        ("pre_fall_vacation", "2020-10-06", 0.6),
        ("fall_vacation", "2020-10-25", 1.0),
        ("post_fall_vacation", "2020-11-01", 0.6),
        ("until_christmas", "2020-12-23", 0.5),
        ("christmas_vacation", "2021-01-10", 0.6),
        ("hard_lockdown", "2021-02-28", 0.45),
        ("before_easter", "2021-04-01", 0.55),
        ("easter_holidays", "2021-04-05", 0.65),
        ("after_easter", VERY_LATE, 0.5),
    ]

    start_date = VERY_EARLY
    to_combine = []
    for prefix, end_date, multiplier in specs:
        block_info = {
            "start_date": start_date,
            "end_date": pd.Timestamp(end_date),
            "prefix": prefix,
        }
        to_combine.append(
            reduce_other_models(
                contact_models=contact_models,
                block_info=block_info,
                multiplier=multiplier,
            ),
        )
        start_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    other_policies = combine_dictionaries(to_combine)
    return other_policies


def _get_enacted_young_educ_policies(contact_models):
    """Get the educ policies for nurseries and preschools.

    The multiplier for the emergency care around the Christmas holidays is
    based on the data for the 1st half of January where many parents might
    have still had vacations.
    1 in 10 primary children attend in emergency care (source only
    for Bavaria, mid January: https://bit.ly/3sGHZbJ). We assume this is similar
    for preschools and nurseries.

    "jan_and_feb_2021":
        Jump from ~25% to 33% between first Jan week and later.
        source: https://bit.ly/3uGL1Pb
        This could be a vacation effect with more parents returning to work.

    "feb_22_to_mid_march":
        We assume nurseries and preschools open normally. This is what happened
        for nurseries in all states with >8 mio inhabitants (BW, BY, NRW) on
        Feb 22nd. (https://bit.ly/3uSp6Ey, https://bit.ly/3h77Cjs,
        https://bit.ly/2O3aS3h)

    "mid_march_to_easter":
        - NRW: preschools and nurseries are open (https://bit.ly/3nSkUBM)
        - BW: emergency care after March 17 (https://bit.ly/3useyeP)
        - BY: emergency care in counties with incidences >100 starting
          March 15 (https://bit.ly/2PRtaW0), the incidence in BY was >100 for
          most of that time frame.

        => assume generous emergency care. This is errs on the side of reducing
           contacts too much.

    "easter_until_april_25":
        - BY: emergency care in counties with incidences >100 (https://bit.ly/3h234eh)
          Incidence was >120 (up to 200) over the whole time
        - BW: open again (https://bit.ly/3h2g83e)
        - NRW: unchanged open (https://bit.ly/33kqof8)

        => assume generous emergency care. This is errs on the side of reducing
           contacts too much.

    "after_april_24" (last updated 2021-05-06):
        - Bundesnotbremse -> emergency care when incidence >165
        - BY (Stand 2021-05-06, https://bit.ly/3xQfJa4): emergency care
          above incidence of 100. State-wide incidence was >130 until May 6.
        - BW (https://bit.ly/3xNNxEF): preschools and nurseries open with
          incidences <165. State-wide incidence was 180 on May 6.
        - NRW: preschools and nurseries open with incidences <165. State-wide
          incidence dropped below 165 on May 2.

        => assume generous emergency care. This is errs on the side of reducing
           contacts too much.

    """
    strict_emergency_care = (
        apply_emergency_care_policies,
        {"attend_multiplier": 0.25, "hygiene_multiplier": HYGIENE_MULTIPLIER},
    )
    generous_emergency_care = (
        apply_emergency_care_policies,
        {"attend_multiplier": 0.34, "hygiene_multiplier": HYGIENE_MULTIPLIER},
    )

    # policies start the day after the end date of the last policy
    specs = [
        ("before_november_2020", "2020-11-01", reduce_educ_models, 1.0),
        ("until_christmas_2020", "2020-12-15", reduce_educ_models, HYGIENE_MULTIPLIER),
        ("christmas-lockdown", "2021-01-10", *strict_emergency_care),
        ("jan_and_feb_2021", "2021-02-21", *generous_emergency_care),
        ("feb_22_to_mid_march", "2021-03-16", reduce_educ_models, HYGIENE_MULTIPLIER),
        ("mid_march_to_easter", "2021-04-05", *generous_emergency_care),
        ("easter_until_april_25", "2021-04-25", *generous_emergency_care),
        ("after_april_24", VERY_LATE, *generous_emergency_care),
    ]

    start_date = VERY_EARLY
    to_combine = []
    for prefix, end_date, func, kwargs in specs:
        block_info = {
            "start_date": start_date,
            "end_date": pd.Timestamp(end_date),
            "prefix": prefix,
        }
        if isinstance(kwargs, float):
            kwargs = {"multiplier": kwargs}

        to_combine.append(
            func(
                contact_models=contact_models,
                block_info=block_info,
                educ_type="young_educ",
                **kwargs,
            )
        )
        start_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    educ_policies = combine_dictionaries(to_combine)
    return educ_policies


def _get_enacted_school_policies(contact_models):
    """Get the policies for schools.

    - "christmas_lockdown":
        This is based on the data for the 1st half of January where many parents
        might have still had vacations. 1 in 10 primary children attend in
        emergency care (source only for BY, mid January: https://bit.ly/3sGHZbJ)
        This abstracts from graduating classes opening early in some states,
        such as Berlin (11 of Jan, https://bit.ly/385iCZk).

    - "jan_and_feb_2021":
        In the second half of January, approx. 1 / 3 of children below secondary
        level were in emergency care

        sources:

        - https://bit.ly/3uGL1Pb
        - https://bit.ly/2PErr5T
        - Berlin: <40% (https://bit.ly/304R5ml)
        - Niedersachsen: 38% (https://bit.ly/2PtPdSb)

        In addition, many states opened graduating classes. We open them at the
        federal level in A / B schooling on Jan 16th. This abstracts from variety:

        - Bavaria: started on 1 Feb (https://bit.ly/3e4p1YE)
        - Baden-Württemberg started on 18 Jan (https://bit.ly/2Ofq9O7)
        - Berlin started on  11 Jan (https://bit.ly/385iCZk)

        sources:

        - https://taz.de/Schulen-in-Coronazeiten/!5753515/
        - https://tinyurl.com/2jfm4tp8

    - "feb_22_to_mid_march":
        Schools open for primary students
            and graduating classes in A/B while maintaining emergency care for children
            with high educ_contact_priority (>0.9 for secondary students <13 and
            >0.66 for primary students).

            Summary of actual policies of states with >=8 mio inhabitants:
                BW: Primary schools open Feb 22nd. Graduating classes attend.
                BY: Primary schools open Feb 22nd. Graduating classes attend.
                NRW: Primary schools and graduating classes start Feb 22nd,
                     A/B or full depending on local incidecne (https://bit.ly/3uSp6Ey)

    - "mid_march_to_easter":
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

        -> We simplify to A/B schooling for everyone plus emergency care
           and graduating classes.

    - "easter_until_may":
        This covers April 6-30. Starting April 26th the Bundesnotbremse is in place.
        That means when counties have a >165 incidence schools only offer
        classes to graduating classes and emergency care. (https://bit.ly/3aUd2KC)
        Since BY has a lower cutoff and cases only fell below 165 around May 1,
        we assume generous emergency care for everyone.

       - BW:
           - source: https://bit.ly/32ABEUr, https://bit.ly/3u6Dcld
           - A/B schooling for graduating classes + 4th grade
           - closures >200 incidence
        - BY:
            - source: https://bit.ly/2QmRNu0, https://bit.ly/32FlgBQ (2021-04-22)
            - incidence <100: A/B for everyone
            - incidence >100: 4th grade and graduating classes in A/B schooling.
              emergency care for rest.
            - mean incidence >130 and increasing over the whole time
        - NRW:
            ⁻ sources: https://bit.ly/3nxGZWb
            - only graduating classes, not in  A/B mode

        => We summarize this as a return to graduating classes in A/B plus
        generous emergceny care (i.e. same as 2nd half of January).

    - "summer_educ_policies":
        Starting April 26th the Bundesnotbremse is in place. That means when counties
        have a >165 incidence schools only offer classes to graduating classes and
        emergency care. (https://bit.ly/3aUd2KC)

        - BW: federal guidelines apply (https://bit.ly/3t7AIBJ,
          https://bit.ly/3aR4yUM)
        - BY:
            - source: https://bit.ly/2RgmsJm (accessed 2021-04-30)
            - incidence <100: A/B for everyone
            - incidence >100: 4th grade and graduating classes in A/B schooling.
              emergency care for rest.
            - incidences >100 in most counties!
        - NRW:
            - sources: https://bit.ly/2QHWChG, https://bit.ly/3gPraZu,
              https://bit.ly/32Zq1q8, https://bit.ly/3nzNhEx
            - A/B schooling when incidences <165 b/c of Bundesnotbremse

        As cases are falling in that time frame (on May 6 ~75% of counties
        were below the 165 threshold, we take the more optimistic scenario that
        schools have A/B schooling for everyone plus emergency care
        and graduating classes.

    """
    strict_emergency_care = _get_school_strict_emergency_care()
    generous_emergency_care = (
        _get_school_generous_emergency_care_with_a_b_for_graduating_classes()
    )
    primary_and_graduating_in_ab = (
        _get_generous_emergency_care_with_a_b_for_primary_and_graduation_classes()
    )
    # A/B for everyone, graduating classes attend in full + emergency care.
    mid_march_to_easter_policy = _get_policy_mid_march_to_easter()

    # combine specs
    specs = [
        ("before_november_2020", "2020-11-01", reduce_educ_models, 1.0),
        ("until_christmas_2020", "2020-12-15", reduce_educ_models, HYGIENE_MULTIPLIER),
        ("christmas-lockdown", "2021-01-10", *strict_emergency_care),
        ("jan_and_feb_2021", "2021-02-21", *generous_emergency_care),
        ("feb_22_to_mid_march", "2021-03-14", *primary_and_graduating_in_ab),
        ("mid_march_to_easter", "2021-04-05", *mid_march_to_easter_policy),
        ("easter_until_may", "2021-04-30", *generous_emergency_care),
        ("summer_educ_policies_2021", VERY_LATE, *mid_march_to_easter_policy),
    ]
    start_date = VERY_EARLY
    to_combine = []
    for prefix, end_date, func, kwargs in specs:
        block_info = {
            "start_date": start_date,
            "end_date": pd.Timestamp(end_date),
            "prefix": prefix,
        }
        if isinstance(kwargs, float):
            kwargs = {"multiplier": kwargs}

        to_combine.append(
            func(
                contact_models=contact_models,
                block_info=block_info,
                educ_type="school",
                **kwargs,
            )
        )
        start_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    school_policies = combine_dictionaries(to_combine)
    return school_policies


def _get_policy_mid_march_to_easter():
    """Get the educ_options starting March 15th until April 5th.

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
        "hygiene_multiplier": HYGIENE_MULTIPLIER,
        # Demand seems to be lower the older the children
        # but only data from Bavaria available: https://bit.ly/3sGHZbJ
        "always_attend_query": always_attend_query,
        # simplify to A/B schooling for everyone
        "a_b_query": "age == age",
        "non_a_b_attend": False,
        # only very anecdotally the current most common a_b_rhythm.
        "a_b_rhythm": "daily",
    }
    return apply_mixed_educ_policies, {"educ_options": educ_options}


def _get_generous_emergency_care_with_a_b_for_primary_and_graduation_classes():
    """Get the school educ_options from February 22nd to March 15th.

    Schools open for primary students and graduating classes in A/B while
    maintaining emergency care for children with high educ_contact_priority
    (>0.9 for secondary students <13 and >0.66 for primary students).

    Summary of actual policies of states with >=8 mio inhabitants:
        BW: primary schools open Feb 22nd. Graduating classes attend.
        BY: primary schools open Feb 22nd. Graduating classes attend.
        NRW: Primaries and graduating classes start Feb 22nd, A/B or full
             depending on local incidence

    Sources:
        - https://www.mdr.de/brisant/corona-schule-geoeffnet-100.html
        - https://bit.ly/2O3aS3h
        - https://bit.ly/3uSp6Ey

    """
    primary_emergency_query = "(educ_contact_priority > 0.66 & age < 10)"
    secondary_emergency_query = "(educ_contact_priority > 0.75 & age >= 10)"
    always_attend_query = f"{primary_emergency_query} | {secondary_emergency_query}"
    educ_options = {
        "hygiene_multiplier": HYGIENE_MULTIPLIER,
        # Demand seems to be lower the older the children
        # but only data from Bavaria available: https://bit.ly/3sGHZbJ
        "always_attend_query": always_attend_query,
        # primary schools and graduating classes in A/B mode
        "a_b_query": "age <= 10 | age >= 16",
        "non_a_b_attend": False,
        # only very anecdotally the current most common a_b_rhythm.
        "a_b_rhythm": "daily",
    }
    return apply_mixed_educ_policies, {"educ_options": educ_options}


def _get_school_generous_emergency_care_with_a_b_for_graduating_classes():
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

    """
    primary_emergency_query = "(educ_contact_priority > 0.66 & age < 10)"
    secondary_emergency_query = "(educ_contact_priority > 0.9 & age >= 10)"
    always_attend_query = f"{primary_emergency_query} | {secondary_emergency_query}"
    educ_options = {
        "hygiene_multiplier": HYGIENE_MULTIPLIER,
        # Demand seems to be lower the older the children
        # but only data from Bavaria available: https://bit.ly/3sGHZbJ
        "always_attend_query": always_attend_query,
        "non_a_b_attend": False,
        # to cover graduating classes
        "a_b_query": "age in [16, 17, 18]",
        # only very anecdotally the current most common a_b_rhythm.
        "a_b_rhythm": "daily",
    }
    return apply_mixed_educ_policies, {"educ_options": educ_options}


def _get_school_strict_emergency_care():
    primary_emergency_query = "(educ_contact_priority > 0.75 & age < 10)"
    secondary_emergency_query = "(educ_contact_priority > 0.95 & age >= 10)"
    always_attend_query = f"{primary_emergency_query} | {secondary_emergency_query}"
    educ_options = {
        "always_attend_query": always_attend_query,
        "a_b_query": False,
        "non_a_b_attend": False,
        "hygiene_multiplier": HYGIENE_MULTIPLIER,
    }
    return apply_mixed_educ_policies, {"educ_options": educ_options}