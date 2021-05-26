import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.shared import create_age_groups
from src.shared import load_dataset


LOCATIONS = [
    "cnt_home",
    "cnt_work",
    "cnt_school",
    "cnt_leisure",
    "cnt_transport",
    "cnt_otherplace",
]


MOSSONG_IN = SRC / "original_data" / "mossong_2008"
MOSSONG_OUT = BLD / "data" / "mossong_2008"


@pytask.mark.depends_on(
    {
        "hh_common": MOSSONG_IN / "hh_common.csv",
        "hh_extra": MOSSONG_IN / "hh_extra.csv",
        "participant_common": MOSSONG_IN / "participant_common.csv",
        "participant_extra": MOSSONG_IN / "participant_extra.csv",
        "contact_common": MOSSONG_IN / "contact_common.csv",
        "sday": MOSSONG_IN / "sday.csv",
        "eu_hh_size_shares": BLD
        / "data"
        / "population_structure"
        / "eu_hh_size_shares.pkl",
        "shared.py": SRC / "shared.py",
    }
)
@pytask.mark.produces(
    {
        "contact_data": MOSSONG_OUT / "contact_data.pkl",
        "hh_sample": MOSSONG_OUT / "hh_sample_ger.csv",
        "hh_probabilities": MOSSONG_OUT / "hh_probabilities.csv",
    }
)
def task_prepare_mossong_data(depends_on, produces):
    datasets = {
        key: load_dataset(val)
        for key, val in depends_on.items()
        if not key.endswith(".py")
    }

    # clean data
    hh = _prepare_hh_data(datasets["hh_common"], datasets["hh_extra"])
    participants = _prepare_participant_data(
        datasets["participant_common"], datasets["participant_extra"]
    )
    contacts = _prepare_contact_data(datasets["contact_common"])
    sday = _prepare_day_data(datasets["sday"])

    # contact_data
    contacts = _merge_mossong_data(
        contacts=contacts, participants=participants, sday=sday, hh=hh
    )
    contacts = _make_columns_in_contact_data_nice(contacts)
    contacts = contacts[contacts["country"].isin(["LU", "DE_TOT", "BE", "NL"])]
    contacts = contacts.dropna(how="any")
    contacts.to_pickle(produces["contact_data"])

    # household sample for initial states
    hh = hh.query("country == 'DE_TOT'")
    hh = _from_wide_to_long_format(hh)
    hh = _drop_hh_with_missing_ages(hh)
    hh.to_csv(produces["hh_sample"])

    # household probability weights
    hh["collapsed_hh_size"] = hh["hh_size"].where(
        hh["hh_size"] <= 5, pd.Interval(5.0, np.inf)
    )
    sample_hh_size_shares = hh["collapsed_hh_size"].value_counts(normalize=True)
    inv_prob_weights = datasets["eu_hh_size_shares"]["DE_TOT"] / sample_hh_size_shares
    hh["hh_inv_prob_weights"] = hh["collapsed_hh_size"].replace(inv_prob_weights)
    hh["probability"] = hh["hh_inv_prob_weights"] / hh["hh_inv_prob_weights"].sum()
    hh_probs = hh[["hh_id", "probability"]]
    hh_probs.to_csv(produces["hh_probabilities"])


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
    df = common.copy(deep=True)
    df["frequency"] = _make_frequencies_categorical(df["frequency_multi"])
    df["phys_contact"].replace({1: True, 2: False}, inplace=True)
    df["duration"] = _make_durations_categorical(df["duration_multi"])

    # the order of the location determines for contacts in more than one context to
    # which they are counted. This affects < 10% of contacts.
    assert (df[LOCATIONS].sum(axis=1) > 1).mean() < 0.1
    df["place"] = df.apply(_create_place, axis=1)

    df = df.rename(columns={loc: loc[4:] for loc in LOCATIONS})
    df.drop(columns=["frequency_multi", "duration_multi"], inplace=True)
    return df


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


def _merge_mossong_data(contacts, participants, hh, sday):
    df = pd.merge(left=contacts, right=participants, on="part_id", validate="m:1")
    df = pd.merge(left=df, right=hh, on="hh_id", validate="m:1")
    df = pd.merge(left=df, right=sday, on="part_id", validate="m:1")
    df.set_index("cont_id", inplace=True)

    # remove problematic entries
    df = df[df["problems"] != "Y"]

    # remove one kid's work contact
    df = df[~((df["part_age"] < 15) & (df["work"]))]
    return df


def _from_wide_to_long_format(hh):
    """Convert the data from wide to long format."""
    # To long format.
    age_columns = [f"hh_age_{x}" for x in range(1, 21)]
    hh = hh.melt(
        id_vars=["hh_id", "country", "hh_size"],
        value_vars=age_columns,
        var_name="p_id",
        value_name="age",
    )

    # Create personal id from the order in which ages were reported.
    hh["p_id"] = hh["p_id"].str.split("_").str[-1].astype(np.uint8)

    # Remove all observations which were artificially created in wide format.
    hh = hh.loc[hh.p_id.le(hh.hh_size)]

    hh = hh.astype({"hh_id": "category", "country": "category"})

    return hh


def _make_columns_in_contact_data_nice(df):
    df = df.copy(deep=True)
    df = df.rename(
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

    # drop columns
    keep_cols = [
        "age_of_contact",
        "country",
        "day",
        "dayofweek",
        "duration",
        "frequency",
        "gender_of_contact",
        "hh_id",
        "hh_size",
        "home",
        "id",
        "leisure",
        "month",
        "otherplace",
        "part_age",
        "part_gender",
        "participant_occupation",
        "phys_contact",
        "place",
        "school",
        "transport",
        "weekend",
        "work",
        "year",
    ]
    df = df[keep_cols]

    # add columns
    df["part_age_group"] = create_age_groups(df["part_age"])
    df["part_broad_age_group"] = pd.cut(df["part_age"], [0, 30, 60, 100])
    df["age_group_of_contact"] = create_age_groups(df["age_of_contact"])
    df["recurrent"] = df["frequency"].isin(["1-2 times a week", "(almost) daily"])
    return df


def _drop_hh_with_missing_ages(df):
    """Drop households that don't have ages for every person in the household."""
    df = df.copy(deep=True)
    df = df.dropna()

    # Keep only complete households.
    n_hh_members_with_age = df.groupby("hh_id")["p_id"].transform("size")
    df = df.loc[df.hh_size.eq(n_hh_members_with_age)]

    # Drop households consisting of children only
    oldest_above_16 = df.groupby("hh_id")["age"].max() >= 16
    adult_led_hh = oldest_above_16[oldest_above_16].index
    df = df[df["hh_id"].isin(adult_led_hh)]

    df["age"] = df["age"].astype(np.uint8)
    return df
