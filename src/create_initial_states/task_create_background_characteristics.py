"""Create a synthetic population that is representative of Germany."""
import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.config import SRC
from src.create_initial_states.create_contact_model_group_ids import (
    add_contact_model_group_ids,
)
from src.shared import create_age_groups
from src.shared import create_age_groups_rki


@pytask.mark.depends_on(
    {
        "py1": SRC / "create_initial_states" / "create_contact_model_group_ids.py",
        "py2": SRC / "create_initial_states" / "add_weekly_ids.py",
        "py3": SRC / "create_initial_states" / "make_educ_group_columns.py",
        "hh_data": SRC
        / "original_data"
        / "population_structure"
        / "microcensus2010_cf.dta",
        "county_probabilities": BLD / "data" / "counties.parquet",
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
@pytask.mark.produces(BLD / "data" / "initial_states_microcensus.parquet")
def task_create_initial_states_microcensus(depends_on, produces):
    mc = pd.read_stata(depends_on["hh_data"])
    mc = _prepare_microcensus(mc)

    equal_probs = pd.DataFrame()
    equal_probs["hh_id"] = mc["hh_id"].unique()
    equal_probs["probability"] = 1 / len(equal_probs)

    df = _sample_mc_hhs(mc, equal_probs, n_households=1_000_000, seed=4874)

    county_probabilities = pd.read_parquet(depends_on["county_probabilities"])
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

    _check_background_characteristics(df)

    repeating_contact_distributions = [
        "work_daily_dist",
        "work_weekly_dist",
        "other_daily_dist",
        "other_weekly_dist",
    ]

    group_id_specs = {
        name: pd.read_pickle(depends_on[name])
        for name in repeating_contact_distributions
    }
    df = add_contact_model_group_ids(
        df,
        **group_id_specs,
        seed=555,
    )
    _check_group_ids(df, **group_id_specs)

    df.index.name = "index"
    df = df.drop(columns=["index", "work_type"])
    df.to_parquet(produces)


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
    }
    mc = mc.rename(columns=rename_dict)
    mc = mc[rename_dict.values()]
    mc["private_hh"] = mc["hh_form"] == "bevölkerung in privathaushalten"
    mc["gender"] = mc["gender"].replace({"männlich": "male", "weiblich": "female"})

    mc["age"] = mc["age"].replace({"95 jahre und älter": 96})
    mc["age_group"] = create_age_groups(mc["age"])
    mc["age_group_rki"] = create_age_groups_rki(mc)

    mc["hh_id"] = mc.apply(_create_mc_hh_id, axis=1)
    mc["hh_id"] = pd.factorize(mc["hh_id"])[0]
    assert len(mc["hh_id"].unique()) == 11_494, "Wrong number of households."
    mc = mc.drop(columns=["district_id", "east_west", "hh_form", "hh_nr_in_district"])
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


def _check_background_characteristics(df):
    """Check that the background characteristics come out right."""
    df = df.copy(deep=True)
    df["female"] = df["gender"] == "female"
    df["male"] = df["gender"] == "male"

    assert df["hh_id"].value_counts().max() <= 42
    assert df["age"].between(0, 110).all()
    assert (40 <= df["age"].median()) & (48 >= df["age"].median())
    assert df["state"].nunique() == 16

    assert df["gender"].value_counts(normalize=True).between(0.48, 0.52).all()
    young_gender_shares = df.query("age < 18")["gender"].value_counts(normalize=True)
    assert young_gender_shares.between(0.48, 0.52).all()
    df.groupby("hh_id")["gender"]
    old_women_share = (df.query("70 < age < 80")["gender"] == "female").mean()
    assert old_women_share > 0.53
    old_women_share = (df.query("age > 80")["gender"] == "female").mean()
    assert old_women_share > 0.62
    assert (df.query("occupation == 'working'")["gender"] == "female").mean() < 0.5
    # given how we assign gender,
    # men and women should appear in much more than half of households
    assert df.groupby("hh_id")["female"].any().mean() > 0.75
    assert df.groupby("hh_id")["male"].any().mean() > 0.75

    occupation_categories = [
        "working",
        "nursery",
        "preschool",
        "school",
        "stays home",
        "retired",
    ]
    assert df["occupation"].isin(occupation_categories).all()
    assert 0.45 < (df["occupation"] == "working").mean() < 0.55
    assert 0.15 < (df["occupation"] == "retired").mean() < 0.25
    assert 0.12 < (df["occupation"] == "stays home").mean() < 0.18
    assert (df[df["age"].between(6, 14)]["occupation"] == "school").all()
    assert (df[df["age"].between(3, 5)]["occupation"] == "preschool").all()
    assert df[df["age"] < 3]["occupation"].isin(["nursery", "stays home"]).all()
    assert 0.33 <= (df[df["age"] < 3]["occupation"] == "nursery").mean() <= 0.38
    assert 0.9 < (df[df["age"] > 70]["occupation"] == "retired").mean()


def _check_group_ids(
    df,
    work_daily_dist,
    work_weekly_dist,
    other_daily_dist,
    other_weekly_dist,
):
    _check_systemically_relevant(df)
    _check_work_contact_priority(df)
    _check_educ_group_ids(df)
    _check_work_group_ids(df, work_daily_dist, work_weekly_dist)
    _check_other_group_ids(df, other_daily_dist, other_weekly_dist)


def _check_systemically_relevant(df):
    shares = df.groupby("occupation")["systemically_relevant"].mean()
    assert shares["retired"] == 0.0, "retired systemically relevant."
    assert shares["school"] == 0.0, "children systemically relevant."
    teach_occs = ["school_teacher", "preschool_teacher", "nursery_teacher"]
    assert (shares[teach_occs] == 1.0).all(), "not all teachers systemically relevant."
    assert shares["stays home"] == 0.0, "stays home systemically relevant."
    assert (0.31 < shares["working"]) and (
        shares["working"] < 0.35
    ), "not a third of workers systemically_relevant"

    not_working = "occupation in ['stays home', 'retired', 'school']"
    assert not df.query(not_working)["systemically_relevant"].any()
    workers = df.query("occupation == 'working'")
    assert workers["systemically_relevant"].mean() > 0.27
    assert workers["systemically_relevant"].mean() < 0.35


def _check_work_contact_priority(df):
    not_working = "occupation in ['stays home', 'retired', 'school']"
    assert (df.query(not_working)["work_contact_priority"] == -1).all()
    assert (df.query("systemically_relevant")["work_contact_priority"] == 2).all()
    non_essential_prios = df.query("occupation == 'working' & ~ systemically_relevant")[
        "work_contact_priority"
    ]
    assert non_essential_prios.between(-0.01, 1.01).all()
    assert non_essential_prios.std() > 0.2
    assert (non_essential_prios.mean() < 0.52) & (non_essential_prios.mean() > 0.48)


def _check_educ_group_ids(df):
    assert df.notnull().all().all(), "No NaN allowed in the initial states."
    assert set(df["occupation"].cat.categories) == {
        "nursery",
        "preschool",
        "school",
        "working",
        "stays home",
        "retired",
        "school_teacher",
        "preschool_teacher",
        "nursery_teacher",
    }

    for age in range(6, 15):
        students = df.query(f"age == {age}")
        pd.testing.assert_series_equal(
            students["school_group_id_0"],
            students["school_group_id_1"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            students["school_group_id_0"],
            students["school_group_id_2"],
            check_names=False,
        )

    assert (df["school_group_a"].unique() == [1, 0]).all()
    assert df.query("age < 3")["occupation"].isin(["nursery", "stays home"]).all()
    assert (df.query("3 <= age <= 14")["nursery_group_id_0"].value_counts() == 1).all()
    assert (df.query("3 <= age < 6")["preschool_group_id_0"].value_counts() > 1).all()
    assert (df.query("6 < age <= 14")["preschool_group_id_0"].value_counts() == 1).all()
    assert (df.query("6 < age <= 14")["school_group_id_0"].value_counts() > 1).all()

    _check_educators(df)
    _check_educ_group_sizes(df)
    _check_educ_group_assortativeness(df)


def _check_educators(df):
    educators = df[df["occupation"].str.endswith("_teacher")]
    assert (educators["age"] >= 25).all()
    assert (educators["age"] <= 75).all()

    # source: https://tinyurl.com/y3psel4p
    pct_teachers = 782_613 / POPULATION_GERMANY
    assert np.abs((df["occupation"] == "school_teacher").mean() - pct_teachers) < 0.004

    # source: https://tinyurl.com/y2v8zlgo
    pct_preschool_teachers = 380_000 / POPULATION_GERMANY
    share_preschool_teachers = (df["occupation"] == "preschool_teacher").mean()
    assert np.abs(share_preschool_teachers - pct_preschool_teachers) < 0.002


def _check_educ_group_sizes(df):
    name_to_class_bounds = {
        # school target is 23 pupils + 2 teachers => 20, 31
        "school": (20, 31, 2),
        # preschool target is 9 pupils + 2 adults => 11 +/- 1
        "preschool": (10, 12, 2),
        # nursery target is 4 pupils + 1 adult => 5 +/- 1
        "nursery": (4, 6, 1),
    }
    for name, (lower, upper, expected_n_teachers) in name_to_class_bounds.items():
        id_col = f"{name}_group_id_0"
        class_id_to_size = df.groupby(id_col).size()
        ids_of_true_classes = class_id_to_size[class_id_to_size > 1].index
        pupils_and_teachers = df[df[id_col].isin(ids_of_true_classes)]
        class_sizes = pupils_and_teachers[id_col].value_counts().unique()
        assert (class_sizes >= lower).all()
        assert (class_sizes <= upper).all()
        n_teachers = pupils_and_teachers.groupby(id_col)["occupation"].apply(
            lambda x: (x == f"{name}_teacher").sum()
        )
        assert (n_teachers == expected_n_teachers).all()


def _check_educ_group_assortativeness(df):
    col_to_limits = {
        "nursery_group_id_0": (3, [2, 3, 4]),
        "preschool_group_id_0": (4, [3, 4, 5]),
        "school_group_id_0": (7, [2, 3]),
    }
    for col, (max_counties, allowed_n_ages) in col_to_limits.items():
        class_id_to_size = df.groupby(col).size()
        ids_of_true_classes = class_id_to_size[class_id_to_size > 1].index
        pupils_and_teachers = df[df[col].isin(ids_of_true_classes)]
        grouped = pupils_and_teachers.groupby(col)
        assert (grouped["state"].nunique() == 1).all()
        assert grouped["county"].nunique().max() <= max_counties
        assert grouped["county"].nunique().mode()[0] == 1
        assert sorted(grouped["age"].nunique().unique()) == allowed_n_ages


def _check_work_group_ids(df, daily_dist, weekly_dist):
    df = df.copy()

    # create helpers
    w_weekly_cols = [x for x in df if x.startswith("work_weekly_group")]
    n_weekly_w_groups = df[w_weekly_cols].replace(-1, np.nan).notnull().sum(axis=1)
    df["n_weekly_w_groups"] = n_weekly_w_groups

    workers = df.query("occupation == 'working'")
    non_workers = df.query("occupation != 'working'")

    # weekly group ids
    assert len(w_weekly_cols) == 14
    assert (non_workers[w_weekly_cols] == -1).all().all()
    w_weekly_size_shares = workers["n_weekly_w_groups"].value_counts(normalize=True)
    assert np.abs(w_weekly_size_shares - weekly_dist).max() < 0.04

    # daily group ids
    w_daily_group_vc = workers["work_daily_group_id"].value_counts()
    w_daily_group_vc = w_daily_group_vc[w_daily_group_vc > 0]
    assert w_daily_group_vc.max() <= 16
    assert (non_workers["work_daily_group_id"] == -1).all()
    assert (workers["work_daily_group_id"] != -1).all()

    # compare true and target distribution (incomplete!)
    w_daily_group_size_shares = w_daily_group_vc.value_counts(normalize=True)
    assert w_daily_group_size_shares[::-1].is_monotonic
    goal_w_daily_group_size_shares = daily_dist.copy(deep=True)
    goal_w_daily_group_size_shares.index += 1
    assert w_daily_group_size_shares.argmax() == goal_w_daily_group_size_shares.argmax()


def _check_other_group_ids(df, daily_dist, weekly_dist):
    df = df.copy()

    o_weekly_cols = [x for x in df if x.startswith("other_weekly_group")]
    n_weekly_o_groups = df[o_weekly_cols].replace(-1, np.nan).notnull().sum(axis=1)
    df["n_weekly_o_groups"] = n_weekly_o_groups
    o_weekly_size_shares = df["n_weekly_o_groups"].value_counts(normalize=True)

    assert len(o_weekly_cols) == 4
    assert np.abs(o_weekly_size_shares - weekly_dist).max() < 0.08
    o_daily_group_vc = df["other_daily_group_id"].value_counts()
    o_daily_group_vc = o_daily_group_vc[o_daily_group_vc > 0]
    assert o_daily_group_vc.max() <= 6
    o_daily_group_size_shares = o_daily_group_vc.value_counts(normalize=True)
    goal_o_daily_group_size_shares = daily_dist.copy(deep=True)
    goal_o_daily_group_size_shares.index += 1
    diff_btw_o_shares = o_daily_group_size_shares - goal_o_daily_group_size_shares
    assert np.abs(diff_btw_o_shares).max() < 0.1
