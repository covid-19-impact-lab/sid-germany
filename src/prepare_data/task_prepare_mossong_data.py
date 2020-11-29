import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.shared import create_age_groups


LOCATIONS = [
    "cnt_home",
    "cnt_work",
    "cnt_school",
    "cnt_leisure",
    "cnt_transport",
    "cnt_otherplace",
]


MOSSONG_PATH = SRC / "original_data" / "mossong_2008"


@pytask.mark.depends_on(
    {
        "hh_common": MOSSONG_PATH / "hh_common.csv",
        "hh_extra": MOSSONG_PATH / "hh_extra.csv",
        "participant_common": MOSSONG_PATH / "participant_common.csv",
        "participant_extra": MOSSONG_PATH / "participant_extra.csv",
        "contact_common": MOSSONG_PATH / "contact_common.csv",
        "sday": MOSSONG_PATH / "sday.csv",
    }
)
@pytask.mark.produces(BLD / "data" / "mossong_2008" / "contact_data.pkl")
def task_prepare_mossong_data(depends_on, produces):
    datasets = {
        key: pd.read_csv(val, low_memory=False) for key, val in depends_on.items()
    }
    hh = _prepare_hh_data(datasets["hh_common"], datasets["hh_extra"])
    participants = _prepare_participant_data(
        datasets["participant_common"], datasets["participant_extra"]
    )
    contacts = _prepare_contact_data(datasets["contact_common"])
    sday = _prepare_day_data(datasets["sday"])

    contacts = pd.merge(left=contacts, right=participants, on="part_id", validate="m:1")
    contacts = pd.merge(left=contacts, right=hh, on="hh_id", validate="m:1")
    contacts = pd.merge(left=contacts, right=sday, on="part_id", validate="m:1")
    contacts.set_index("cont_id", inplace=True)

    # remove problematic entries
    contacts = contacts[contacts["problems"] != "Y"]

    contacts = contacts.rename(
        columns={
            "cnt_age_exact": "age_of_contact",
            "cnt_gender": "gender_of_contact",
            "duration_multi": "duration",
            "frequency_multi": "frequency",
            "part_education_length": "participant_edu",
            "part_id": "id",
            "part_occupation": "participant_occupation",
        }
    )

    drop_cols = [
        "child_care",
        "child_care_detail",
        "problems",
        "participant_school_year",
        "part_occupation_detail",
        "child_nationality",
        "part_education",
        "child_relationship",
        "cnt_age_est_max",
        "cnt_age_est_min",
        "diary_how",
        "type",
    ]
    drop_cols += [f"hh_age_{x}" for x in range(5, 21)]  # noqa
    contacts.drop(columns=drop_cols, inplace=True)

    contacts["part_age_group"] = create_age_groups(contacts["part_age"])
    contacts["age_group_of_contact"] = create_age_groups(contacts["age_of_contact"])

    contacts["recurrent"] = contacts["frequency"].isin(
        ["1-2 times a week", "(almost) daily"]
    )

    # remove one kid's work contact
    contacts = contacts[~((contacts["part_age"] < 15) & (contacts["work"]))]
    contacts.to_pickle(produces)


def _prepare_hh_data(common, extra):
    common = common.copy()
    common["country"] = common["country"].replace({"DE": "DE_TOT", "GB": "UK"})
    hh = pd.merge(left=common, right=extra, on="hh_id")
    return hh


def _prepare_participant_data(common, extra):
    common = common.copy(deep=True)
    extra = extra.copy(deep=True)
    extra["part_occupation"].replace(
        {
            1: "working",
            2: "retired",
            3: "at home (housewife)",
            4: "unemployed",
            5: "fulltime education",
            6: "other",
        },
        inplace=True,
    )

    missed_d = {1: 0, 2: "1-4", 3: "5-9", 4: ">10"}
    rename = [
        ("nr_missed_to_record", "diary_missed_unsp"),
        ("nr_missed_to_record_physical", "diary_missed_skin"),
        ("nr_missed_to_record_not_physical", "diary_missed_noskin"),
    ]
    for new, old in rename:
        extra[new] = extra[old].replace(missed_d)
        extra.drop(columns=[old], inplace=True)

    participants = pd.merge(left=common, right=extra, on="part_id")
    return participants


def _prepare_contact_data(common):

    contacts = common.copy(deep=True)
    contacts["frequency"] = _make_frequencies_categorical(contacts["frequency_multi"])
    contacts["phys_contact"].replace({1: True, 2: False}, inplace=True)
    contacts["duration"] = _make_durations_categorical(contacts["duration_multi"])

    # the order of the location determines for contacts in more than one context to
    # which they are counted. This affects < 10% of contacts.
    assert (contacts[LOCATIONS].sum(axis=1) > 1).mean() < 0.1
    contacts["place"] = contacts.apply(_create_place, axis=1)

    contacts = contacts.rename(columns={loc: loc[4:] for loc in LOCATIONS})
    contacts.drop(columns=["frequency_multi", "duration_multi"], inplace=True)
    return contacts


def _make_frequencies_categorical(sr):
    rename_dict = {
        1: "(almost) daily",
        2: "1-2 times a week",
        3: "1-2 times a month",
        4: "less than once a month",
        5: "never met before",
    }
    nice_sr = sr.replace(rename_dict)
    frequencies = [
        "(almost) daily",
        "1-2 times a week",
        "1-2 times a month",
        "less than once a month",
        "never met before",
    ]
    return pd.Categorical(nice_sr, categories=frequencies, ordered=True)


def _make_durations_categorical(sr):
    durations = ["<5min", "5-15min", "15min-1h", "1-4h", ">4h"]
    rename_dict = {
        1: "<5min",
        2: "5-15min",
        3: "15min-1h",
        4: "1-4h",
        5: ">4h",
    }
    nice_sr = sr.replace(rename_dict)
    return pd.Categorical(nice_sr, categories=durations, ordered=True)


def _create_place(row):
    for loc in LOCATIONS:
        if row[loc]:
            return loc[4:]


def _prepare_day_data(sday):
    sday = sday.copy(deep=True)
    sday["dayofweek"].replace(
        {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"},
        inplace=True,
    )
    sday["dayofweek"] = pd.Categorical(
        sday["dayofweek"],
        categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ordered=True,
    )
    sday["weekend"] = sday["dayofweek"].isin(["Sat", "Sun"])
    return sday
