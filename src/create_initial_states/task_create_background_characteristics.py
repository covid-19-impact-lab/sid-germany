"""Create a synthetic population that is representative of Germany."""
import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import N_HOUSEHOLDS
from src.config import SHARE_REFUSE_VACCINATION
from src.config import SRC
from src.create_initial_states.create_contact_model_group_ids import (
    add_contact_model_group_ids,
)
from src.create_initial_states.create_vaccination_priority import (
    create_vaccination_group,
)
from src.create_initial_states.create_vaccination_priority import (
    create_vaccination_rank,
)
from src.shared import create_age_groups
from src.shared import create_age_groups_rki


@pytask.mark.depends_on(
    {
        "py1": SRC / "create_initial_states" / "create_contact_model_group_ids.py",
        "py2": SRC / "create_initial_states" / "add_weekly_ids.py",
        "py3": SRC / "create_initial_states" / "make_educ_group_columns.py",
        "py4": SRC / "create_initial_states" / "create_vaccination_priority.py",
        "hh_data": SRC
        / "original_data"
        / "population_structure"
        / "microcensus2010_cf.dta",
        "county_probabilities": BLD
        / "data"
        / "population_structure"
        / "counties.parquet",
        "work_daily_dist": BLD
        / "contact_models"
        / "empirical_distributions"
        / "work_recurrent_daily.pkl",
        "work_weekly_dist": BLD
        / "contact_models"
        / "empirical_distributions"
        / "work_recurrent_weekly.pkl",
        "other_daily_dist": BLD
        / "contact_models"
        / "empirical_distributions"
        / "other_recurrent_daily.pkl",
        "other_weekly_dist": BLD
        / "contact_models"
        / "empirical_distributions"
        / "other_recurrent_weekly.pkl",
    }
)
@pytask.mark.produces(
    {
        N_HOUSEHOLDS: BLD / "data" / "initial_states.parquet",
        100_000: BLD / "data" / "debug_initial_states.parquet",
    }
)
def task_create_initial_states_microcensus(depends_on, produces):
    mc = pd.read_stata(depends_on["hh_data"])
    county_probabilities = pd.read_parquet(depends_on["county_probabilities"])
    work_daily_dist = pd.read_pickle(depends_on["work_daily_dist"])
    work_weekly_dist = pd.read_pickle(depends_on["work_weekly_dist"])
    other_daily_dist = pd.read_pickle(depends_on["other_daily_dist"])
    other_weekly_dist = pd.read_pickle(depends_on["other_weekly_dist"])

    for n_hhs, path in produces.items():
        df = _build_initial_states(
            mc=mc,
            county_probabilities=county_probabilities,
            work_daily_dist=work_daily_dist,
            work_weekly_dist=work_weekly_dist,
            other_daily_dist=other_daily_dist,
            other_weekly_dist=other_weekly_dist,
            n_households=n_hhs,
            seed=4874,
        )
        df.to_parquet(path)


def _build_initial_states(
    mc,
    county_probabilities,
    work_daily_dist,
    work_weekly_dist,
    other_daily_dist,
    other_weekly_dist,
    n_households,
    seed,
):
    mc = _prepare_microcensus(mc)

    equal_probs = pd.DataFrame()
    equal_probs["hh_id"] = mc["hh_id"].unique()
    equal_probs["probability"] = 1 / len(equal_probs)

    df = _sample_mc_hhs(mc, equal_probs, n_households=n_households, seed=seed)

    county_and_state = _draw_counties(
        hh_ids=df["hh_id"].unique(),
        county_probabilities=county_probabilities,
        seed=2282,
    )
    df = df.merge(county_and_state, on="hh_id", validate="m:1")
    df = df.astype({"age": np.uint8, "hh_id": "category"})
    df = df.sort_values("hh_id").reset_index()
    df.index.name = "temp_index"

    assert not df.index.duplicated().any()
    df["occupation"] = _create_occupation(df)

    df = add_contact_model_group_ids(
        df,
        work_daily_dist=work_daily_dist,
        work_weekly_dist=work_weekly_dist,
        other_daily_dist=other_daily_dist,
        other_weekly_dist=other_weekly_dist,
        seed=555,
    )

    adult_at_home = (df["occupation"].isin(["stays home", "retired"])) & (
        df["age"] >= 18
    )
    df["adult_in_hh_at_home"] = adult_at_home.groupby(df["hh_id"]).transform(any)
    df["educ_contact_priority"] = _create_educ_contact_priority(df)

    df["vaccination_group"] = create_vaccination_group(states=df, seed=484)
    df["vaccination_rank"] = create_vaccination_rank(
        df["vaccination_group"], share_refuser=SHARE_REFUSE_VACCINATION, seed=909
    )

    df.index.name = "index"
    df = _only_keep_relevant_columns(df)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def _prepare_microcensus(mc):
    rename_dict = {
        "ef1": "east_west",
        "ef3s": "district_id",
        "ef4s": "hh_nr_in_district",
        "ef20": "hh_size",
        "ef29": "work_type",
        "ef31": "hh_form",
        "ef44": "age",
        "ef46": "gender",
        "ef149": "frequency_work_saturday",
        "ef150": "frequency_work_sunday",
    }
    mc = mc.rename(columns=rename_dict)
    mc = mc[rename_dict.values()]
    mc["private_hh"] = mc["hh_form"] == "bevölkerung in privathaushalten"
    # restrict to private households for the moment
    mc = mc[mc["private_hh"]]
    mc["gender"] = (
        mc["gender"]
        .replace({"männlich": "male", "weiblich": "female"})
        .astype("category")
    )

    mc["age"] = mc["age"].replace({"95 jahre und älter": 96})
    mc["age_group"] = create_age_groups(mc["age"])
    mc["age_group_rki"] = create_age_groups_rki(mc)

    # 53% no, 21% every now and then, 17% regularly, 9% all the time
    work_answers = ["ja, ständig", "ja, regelmäßig"]
    mc["work_saturday"] = mc["frequency_work_saturday"].isin(work_answers)
    # 72% no, 14% every now and then, 10% regularly, 3% all the time
    mc["work_sunday"] = mc["frequency_work_sunday"].isin(work_answers)

    mc["hh_id"] = mc.apply(_create_mc_hh_id, axis=1)
    mc["hh_id"] = pd.factorize(mc["hh_id"])[0]
    assert len(mc["hh_id"].unique()) == 11_461, "Wrong number of households."

    keep_cols = [
        "private_hh",
        "gender",
        "age",
        "age_group",
        "age_group_rki",
        "work_type",
        "work_saturday",
        "work_sunday",
        "hh_id",
    ]
    mc = mc[keep_cols]
    return mc


def _create_mc_hh_id(row):
    hh_id_parts = ["east_west", "district_id", "hh_nr_in_district"]
    row_id = "_".join(str(row[var]) for var in hh_id_parts)
    return row_id


def _sample_mc_hhs(mc, hh_probabilities, n_households, seed):
    np.random.seed(seed)
    sampled_ids = np.random.choice(
        hh_probabilities.hh_id,
        p=hh_probabilities.probability,
        size=n_households,
        replace=True,
    )
    new_id_df = pd.DataFrame({"old_hh_id": sampled_ids})
    new_id_df = new_id_df.reset_index()
    new_id_df = new_id_df.rename(columns={"index": "hh_id"})

    df = new_id_df.merge(
        mc,
        left_on="old_hh_id",
        right_on="hh_id",
        validate="m:m",
        suffixes=("", "_"),
    )
    df = df.drop(columns=["old_hh_id", "hh_id_"])
    df = df.sort_values("hh_id")
    df["hh_id"] = df["hh_id"].astype("category")
    df = df.reset_index(drop=True)
    return df


def _draw_counties(hh_ids, county_probabilities, seed):
    """Draw for each household to which county and federal state it belongs to."""
    np.random.seed(seed)
    sampled_counties = np.random.choice(
        county_probabilities.id,
        p=county_probabilities.weight,
        size=len(hh_ids),
        replace=True,
    )
    df = pd.DataFrame({"county": sampled_counties})
    df = df.reset_index()
    df = df.rename(columns={"index": "hh_id"})
    df = df.merge(
        county_probabilities[["id", "state"]], left_on="county", right_on="id"
    )
    df = df.drop(columns="id")

    translate_state_names = {
        "Bayern": "Bavaria",
        "Niedersachsen": "Lower Saxony",
        "Nordrhein-Westfalen": "North Rhine-Westphalia",
        "Rheinland-Pfalz": "Rhineland-Palatinate",
        "Sachsen": "Saxony",
        "Sachsen-Anhalt": "Saxony-Anhalt",
        "Thüringen": "Thuringia",
    }
    df["state"] = df["state"].replace(translate_state_names)
    df = df.astype({"state": "category", "county": "category"})
    return df


def _create_occupation(df):
    occupation = pd.Series(np.nan, index=df.index)
    occupation = occupation.where(df["work_type"] != "erwerbstätige", other="working")

    to_fill_nans = pd.Series(np.nan, index=df.index)
    to_fill_nans[df["age"] > 60] = "retired"
    # between is inclusive by default, i.e. lower <= sr <= upper
    to_fill_nans[df["age"].between(6, 19)] = "school"
    to_fill_nans[df["age"].between(3, 5)] = "preschool"

    below_3 = df.query("age < 3").index
    share_of_children_in_nursery = 0.35
    n_to_draw = int(share_of_children_in_nursery * len(below_3))
    attend_nursery_indices = np.random.choice(below_3, size=n_to_draw, replace=False)
    to_fill_nans[attend_nursery_indices] = "nursery"
    to_fill_nans = to_fill_nans.fillna("stays home")
    occupation = occupation.fillna(to_fill_nans).astype("category")
    return occupation


def _create_educ_contact_priority(df):
    """Create an educ contact priority Series.

    The values are 0 for anyone above 12 and uniform from 0 to 1 below.
    Children in households where no adult is retired or stays home have a
    higher educ contact priority than chidldren where at least one adult is
    at home.

    """
    high_priority_children = df.eval("age < 13 & ~adult_in_hh_at_home")
    share_high_priority_children = high_priority_children[df["age"] < 13].mean()
    threshold = 1 - share_high_priority_children

    educ_contact_priority = pd.Series(0, index=df.index)
    low_priorities = np.random.uniform(low=0, high=threshold, size=len(df))
    high_priorities = np.random.uniform(low=threshold, high=1, size=len(df))
    not_low_priority_children = high_priority_children | (df["age"] >= 13)
    educ_contact_priority = educ_contact_priority.where(
        ~high_priority_children, high_priorities
    )
    educ_contact_priority = educ_contact_priority.where(
        not_low_priority_children, low_priorities
    )
    return educ_contact_priority


def _only_keep_relevant_columns(df):
    background_vars = [
        "age",  # used by `implement_a_b_school_system_above_age`
        "age_group",  # assort_by variable
        "age_group_rki",  # for plotting and comparison
        "county",  # assort_by variable
        "educ_worker",  # used by `implement_a_b_school_system_above_age`
        "hh_id",  # not really used anywhere but I would still keep it.
        "occupation",
        "private_hh",  # will become relevant if we include nursery homes
        "state",  # needed for school vacations
        "work_contact_priority",
        "work_saturday",
        "work_sunday",
        "hh_model_group_id",
        "adult_in_hh_at_home",
        "educ_contact_priority",
        "vaccination_group",
        "vaccination_rank",
    ]

    educ_contact_group_ids = [
        "nursery_group_id_0",
        "preschool_group_id_0",
        "school_group_id_0",
        "school_group_id_1",
        "school_group_id_2",
    ]
    a_b_vars = [var + "_a_b" for var in educ_contact_group_ids]

    non_educ_contact_group_ids = (
        [
            "other_daily_group_id",
            "work_daily_group_id",
        ]
        + [f"other_weekly_group_id_{i}" for i in range(4)]
        + [f"work_weekly_group_id_{i}" for i in range(14)]
    )

    keep = (
        background_vars + educ_contact_group_ids + a_b_vars + non_educ_contact_group_ids
    )

    to_drop = [
        "gender",
        "index",
        "stays_home_when_schools_close",  # not used at the moment
        "work_type",
    ]

    assert set(keep + to_drop) == set(df.columns)
    return df[keep]
