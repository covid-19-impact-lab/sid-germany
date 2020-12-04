from functools import partial

from src.create_contact_models import contact_model_functions as cm_funcs


def get_household_contact_model():
    household_contact_model = {
        "households": {
            "is_recurrent": True,
            "model": cm_funcs.meet_hh_members,
            "assort_by": ["hh_id"],
            "loc": "households",
        },
    }
    return household_contact_model


# ----------------------------------------------------------------------------
# Education Contact Models
# ----------------------------------------------------------------------------


def get_school_contact_model():
    school_contact_model = {
        "educ_school": {
            "is_recurrent": True,
            "model": partial(cm_funcs.attends_educational_facility, facility="school"),
            "assort_by": ["school_class_id"],
        }
    }
    return school_contact_model


def get_preschool_contact_model():
    preschool_contact_model = {
        "educ_preschool": {
            "is_recurrent": True,
            "model": partial(
                cm_funcs.attends_educational_facility, facility="preschool"
            ),
            "assort_by": ["preschool_group_id"],
        }
    }
    return preschool_contact_model


def get_nursery_contact_model():
    nursery_contact_model = {
        "educ_nursery": {
            "is_recurrent": True,
            "model": partial(cm_funcs.attends_educational_facility, facility="nursery"),
            "assort_by": ["nursery_group_id"],
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
            "assort_by": ["daily_work_group_id"],
            "model": cm_funcs.go_to_work,
            "loc": "work_recurrent_daily",
        },
    }
    return work_daily_contact_model


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
            "assort_by": ["daily_other_group_id"],
            "model": cm_funcs.meet_daily_other_contacts,
        }
    }
    return other_daily_contact_model
