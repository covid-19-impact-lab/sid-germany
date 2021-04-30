"""Educ Options for different phases of 2021."""


def get_educ_options_1st_half_april(
    young_children_multiplier=0.8, school_hygiene_multiplier=0.5
):
    """Get the educ options starting April 6-18.

    Situation:
       - BW:
           - source: https://bit.ly/32ABEUr, https://bit.ly/3u6Dcld
           - A/B schooling for graduating classes + 4th grade
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
    generous emergceny care (~ 2nd half of January).

    """
    return graduating_classes_in_a_b_plus_generous_emergency_care(
        young_children_multiplier=young_children_multiplier,
        school_hygiene_multiplier=school_hygiene_multiplier,
    )


def get_educ_options_mid_march_to_easter(hygiene_multiplier=0.5):
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


def get_educ_options_feb_22_to_march_15(hygiene_multiplier=0.5):
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


def graduating_classes_in_a_b_plus_generous_emergency_care(
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
