from functools import partial

from src.contact_models import contact_model_functions as cm_funcs


def get_all_contact_models():
    contact_models = {
        **get_household_contact_model(),
        # education
        **get_school_contact_models(),
        **get_preschool_contact_model(),
        **get_nursery_contact_model(),
        # work
        **get_work_non_recurrent_contact_model(),
        **get_work_daily_contact_model(),
        **get_work_weekly_contact_models(),
        # other
        **get_other_non_recurrent_contact_model(),
        **get_other_daily_contact_model(),
        **get_other_weekly_contact_models(),
    }
    return contact_models


def get_household_contact_model():
    household_contact_model = {
        "households": {
            "is_recurrent": True,
            "model": partial(cm_funcs.meet_hh_members, seed=334),
            "assort_by": ["hh_id"],
            "loc": "households",
        },
    }
    return household_contact_model


# ----------------------------------------------------------------------------
# Education Contact Models
# ----------------------------------------------------------------------------


def get_school_contact_models():
    school_contact_models = {
        "educ_school_0": {
            "is_recurrent": True,
            "model": partial(
                cm_funcs.attends_educational_facility,
                id_column="school_group_id_0",
                seed=11,
            ),
            "assort_by": ["school_group_id_0"],
        },
        "educ_school_1": {
            "is_recurrent": True,
            "model": partial(
                cm_funcs.attends_educational_facility,
                id_column="school_group_id_1",
                seed=22,
            ),
            "assort_by": ["school_group_id_1"],
        },
        "educ_school_2": {
            "is_recurrent": True,
            "model": partial(
                cm_funcs.attends_educational_facility,
                id_column="school_group_id_2",
                seed=33,
            ),
            "assort_by": ["school_group_id_2"],
        },
    }
    return school_contact_models


def get_preschool_contact_model():
    preschool_contact_model = {
        "educ_preschool": {
            "is_recurrent": True,
            "model": partial(
                cm_funcs.attends_educational_facility,
                id_column="preschool_group_id_0",
                seed=44,
            ),
            "assort_by": ["preschool_group_id_0"],
        }
    }
    return preschool_contact_model


def get_nursery_contact_model():
    nursery_contact_model = {
        "educ_nursery": {
            "is_recurrent": True,
            "model": partial(
                cm_funcs.attends_educational_facility,
                id_column="nursery_group_id_0",
                seed=55,
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
                seed=66,
            ),
        }
    }
    return work_non_recurrent_contact_model


def get_work_daily_contact_model():
    work_daily_contact_model = {
        "work_recurrent_daily": {
            "is_recurrent": True,
            "assort_by": ["work_daily_group_id"],
            "model": partial(cm_funcs.go_to_work, seed=122),
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
        policy = {
            "is_recurrent": True,
            "assort_by": [col_name],
            "model": partial(
                cm_funcs.go_to_weekly_meeting,
                day_of_week=weekdays[n % len(weekdays)],
                group_col_name=col_name,
                seed=77,
            ),
            "loc": model_name,
        }
        work_weekly_contact_models[model_name] = policy
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
                seed=77,
            ),
        }
    }
    return other_non_recurrent_contact_model


def get_other_daily_contact_model():
    other_daily_contact_model = {
        "other_recurrent_daily": {
            "is_recurrent": True,
            "assort_by": ["other_daily_group_id"],
            "model": partial(cm_funcs.meet_daily_other_contacts, seed=99),
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
        policy = {
            "is_recurrent": True,
            "assort_by": [col_name],
            "model": partial(
                cm_funcs.go_to_weekly_meeting,
                day_of_week=days[n % len(days)],
                group_col_name=col_name,
                seed=3474,
            ),
            "loc": model_name,
        }
        other_weekly_contact_models[model_name] = policy
    return other_weekly_contact_models
