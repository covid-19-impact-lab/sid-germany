from functools import partial

import pandas as pd

from src.contact_models import contact_model_functions as cm_funcs
from src.policies.policy_tools import combine_dictionaries


def get_all_contact_models(christmas_mode, n_extra_contacts_before_christmas):
    """Create the full set of contact models.

    Args:
        christmas_mode (str): one of "full", "same_group", "meet_twice".
            - If "full", every household meets with a different set of
              two other households on every of the three holidays.
            - If "same_group", every household meets the same two other
              households on every of the three holidays.
            - If "meet_twice", every household meets the same two other
              households but only once on the 24th and 25th of December.
            - If None, no Christmas models are included.
        n_extra_contacts_before_christmas (float): Number of additional
            contacts before Christmas to cover things like holiday
            shopping and travel.

    Returns:
        contact_models (dict): sid contact model dictionary.

    """

    to_combine = [
        get_household_contact_model(),
        # education
        get_school_contact_models(),
        get_preschool_contact_model(),
        get_nursery_contact_model(),
        # work
        get_work_non_recurrent_contact_model(),
        get_work_daily_contact_model(),
        get_work_weekly_contact_models(),
        # other
        get_other_non_recurrent_contact_model(),
        get_other_daily_contact_model(),
        get_other_weekly_contact_models(),
    ]
    if christmas_mode is not None:
        to_combine.append(
            get_christmas_contact_models(
                christmas_mode, n_extra_contacts_before_christmas
            )
        )
    contact_models = combine_dictionaries(to_combine)
    return contact_models


def get_household_contact_model():
    household_contact_model = {
        "households": {
            "is_recurrent": True,
            "model": cm_funcs.meet_hh_members,
            "assort_by": ["hh_model_group_id"],
            "loc": "households",
        },
    }
    return household_contact_model


# ----------------------------------------------------------------------------
# Education Contact Models
# ----------------------------------------------------------------------------


def get_school_contact_models():
    model_list = []
    for i in range(3):
        model = {
            f"educ_school_{i}": {
                "is_recurrent": True,
                "model": partial(
                    cm_funcs.attends_educational_facility,
                    id_column=f"school_group_id_{i}",
                ),
                "assort_by": [f"school_group_id_{i}"],
            },
        }
        model_list.append(model)
    school_contact_models = combine_dictionaries(model_list)
    return school_contact_models


def get_preschool_contact_model():
    preschool_contact_model = {
        "educ_preschool_0": {
            "is_recurrent": True,
            "model": partial(
                cm_funcs.attends_educational_facility,
                id_column="preschool_group_id_0",
            ),
            "assort_by": ["preschool_group_id_0"],
        }
    }
    return preschool_contact_model


def get_nursery_contact_model():
    nursery_contact_model = {
        "educ_nursery_0": {
            "is_recurrent": True,
            "model": partial(
                cm_funcs.attends_educational_facility,
                id_column="nursery_group_id_0",
            ),
            "assort_by": ["nursery_group_id_0"],
        },
    }
    return nursery_contact_model


# ----------------------------------------------------------------------------
# Work Contact Models
# ----------------------------------------------------------------------------


def get_work_non_recurrent_contact_model():
    work_non_recurrent_contact_model = {
        "work_non_recurrent": {
            "assort_by": ["age_group", "county"],
            "is_recurrent": False,
            "loc": "work_non_recurrent",
            "model": partial(
                cm_funcs.calculate_non_recurrent_contacts_from_empirical_distribution,
                on_weekends=False,
                query="occupation == 'working'",
            ),
        }
    }
    return work_non_recurrent_contact_model


def get_work_daily_contact_model():
    work_daily_contact_model = {
        "work_recurrent_daily": {
            "is_recurrent": True,
            "assort_by": ["work_daily_group_id"],
            "model": cm_funcs.go_to_work,
            "loc": "work_recurrent_daily",
        },
    }
    return work_daily_contact_model


def get_work_weekly_contact_models():
    prefix = "work_weekly_group_id"
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    work_weekly_contact_models = {}
    for n in range(14):
        col_name = f"{prefix}_{n}"
        model_name = f"work_recurrent_weekly_{n}"
        model = {
            "is_recurrent": True,
            "assort_by": [col_name],
            "model": partial(
                cm_funcs.go_to_weekly_meeting,
                day_of_week=weekdays[n % len(weekdays)],
                group_col_name=col_name,
            ),
            "loc": model_name,
        }
        work_weekly_contact_models[model_name] = model
    return work_weekly_contact_models


# ----------------------------------------------------------------------------
# Other Contact Models
# ----------------------------------------------------------------------------


def get_other_non_recurrent_contact_model():
    other_non_recurrent_contact_model = {
        "other_non_recurrent": {
            "is_recurrent": False,
            "loc": "other_non_recurrent",
            "assort_by": ["age_group", "county"],
            "model": partial(
                cm_funcs.calculate_non_recurrent_contacts_from_empirical_distribution,
                on_weekends=True,
                query=None,
            ),
        }
    }
    return other_non_recurrent_contact_model


def get_other_daily_contact_model():
    other_daily_contact_model = {
        "other_recurrent_daily": {
            "is_recurrent": True,
            "loc": "other_recurrent_daily",
            "assort_by": ["other_daily_group_id"],
            "model": partial(
                cm_funcs.meet_daily_other_contacts,
                group_col_name="other_daily_group_id",
            ),
        }
    }
    return other_daily_contact_model


def get_other_weekly_contact_models():
    prefix = "other_weekly_group_id"
    days = [
        "Saturday",
        "Sunday",
        "Tuesday",
        "Thursday",
        "Monday",
        "Friday",
        "Wednesday",
    ]
    other_weekly_contact_models = {}
    for n in range(4):
        col_name = f"{prefix}_{n}"
        model_name = f"other_recurrent_weekly_{n}"
        model = {
            "is_recurrent": True,
            "assort_by": [col_name],
            "model": partial(
                cm_funcs.go_to_weekly_meeting,
                day_of_week=days[n % len(days)],
                group_col_name=col_name,
            ),
            "loc": model_name,
        }
        other_weekly_contact_models[model_name] = model
    return other_weekly_contact_models


def get_christmas_contact_models(mode, n_contacts_before):
    """Create the Christmas contact models.

    Args:
        mode (str): one of "full", "same_group", "meet_twice".
            - If "full", every household meets with a different set of
              two other households on every of the three holidays.
            - If "same_group", every household meets the same two other
              households on every of the three holidays.
            - If "meet_twice", every household meets the same
              two other households but only once on the 24th and 25th
              of December.
        n_contacts_before (float): number of contacts people meet
            before Christmas

    """
    assert isinstance(
        n_contacts_before, (float, int)
    ), "n_contacts_before must be an int or float."
    dates = pd.date_range("2020-12-24", "2020-12-26")

    christmas_contact_models = {
        "holiday_preparation": {
            "is_recurrent": False,
            "loc": "holiday_preparation",
            "assort_by": ["age_group", "county"],
            "model": partial(
                cm_funcs.holiday_preparation_contacts,
                n_contacts=n_contacts_before,
            ),
        }
    }

    if mode == "full":
        for i, date in enumerate(dates):
            model_name = f"christmas_{mode}_{i}"
            col_name = f"christmas_group_id_{i}"
            contact_model = {
                "is_recurrent": True,
                "loc": model_name,
                "assort_by": [col_name],
                "model": partial(
                    cm_funcs.meet_on_holidays, group_col=col_name, dates=[date]
                ),
            }
            christmas_contact_models[model_name] = contact_model

    elif mode == "same_group":
        model_name = "christmas_same_group"
        col_name = "christmas_group_id_0"
        contact_model = {
            "is_recurrent": True,
            "loc": model_name,
            "assort_by": [col_name],
            "model": partial(
                cm_funcs.meet_on_holidays, group_col=col_name, dates=dates
            ),
        }
        christmas_contact_models[model_name] = contact_model

    elif mode == "meet_twice":
        model_name = "christmas_meet_twice"
        col_name = "christmas_group_id_0"
        contact_model = {
            "is_recurrent": True,
            "loc": model_name,
            "assort_by": [col_name],
            "model": partial(
                cm_funcs.meet_on_holidays, group_col=col_name, dates=dates[:2]
            ),
        }
        christmas_contact_models[model_name] = contact_model
    else:
        raise NotImplementedError(
            "Your mode is not one of 'full', 'same_group', 'meet_twice'"
        )

    return christmas_contact_models
