"""Create a synthetic population that is representative of Germany."""
import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.config import SRC
from src.create_initial_states.create_background_characteristics import (
    create_background_characteristics,
)
from src.create_initial_states.create_contact_model_group_ids import (
    add_contact_model_group_ids,
)
from src.shared import load_dataset

DEPENDENCIES = {
    # modules
    "py1": SRC / "create_initial_states" / "create_background_characteristics.py",
    "py2": SRC / "create_initial_states" / "create_contact_model_group_ids.py",
    "py3": SRC / "create_initial_states" / "add_weekly_ids.py",
    "py4": SRC / "create_initial_states" / "make_educ_group_columns.py",
    # data
    "hh_data": BLD / "data" / "mossong_2008" / "hh_sample_ger.csv",
    "hh_probabilities": BLD / "data" / "mossong_2008" / "hh_probabilities.csv",
    "working_probabilities": BLD
    / "data"
    / "population_structure"
    / "working_shares.pkl",
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


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.produces(BLD / "data" / "inital_states.pickle")
def task_create_background_characteristics(depends_on, produces):
    data = {}
    for name, path in depends_on.items():
        if path.suffix != ".py":
            data[name] = load_dataset(path)

    repeating_contact_distributions = [
        "work_daily_dist",
        "work_weekly_dist",
        "other_daily_dist",
        "other_weekly_dist",
    ]
    group_id_specs = {name: data.pop(name) for name in repeating_contact_distributions}

    df = create_background_characteristics(n_households=400_000, seed=3489, **data)
    _check_background_characteristics(df)

    df = add_contact_model_group_ids(
        df,
        **group_id_specs,
        seed=1109,
    )
    _check_group_ids(df, **group_id_specs)
    df.to_pickle(produces)


def _check_background_characteristics(df):
    """Check that the background characteristics come out right."""
    df = df.copy(deep=True)
    df["female"] = df["gender"] == "female"
    df["male"] = df["gender"] == "male"

    assert df["hh_id"].value_counts().max() <= 8
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

    occupation_categories = ["working", "school", "stays home", "retired"]
    assert df["occupation"].isin(occupation_categories).all()
    assert 0.5 < (df["occupation"] == "working").mean()
    assert 0.6 > (df["occupation"] == "working").mean()
    assert 0.08 < (df["occupation"] == "retired").mean()
    assert 0.12 > (df["occupation"] == "retired").mean()
    assert 0.12 < (df["occupation"] == "stays home").mean()
    assert 0.18 > (df["occupation"] == "stays home").mean()
    assert (df[df["age"].between(6, 14)]["occupation"] == "school").all()
    assert (df[df["age"] > 70]["occupation"] == "retired").all()


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
    set(df["occupation"].cat.categories) == {
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

    _check_educators(df)
    _check_educ_group_sizes(df)
    _check_educ_group_assortativeness(df)


def _check_educators(df):
    educators = df[df["occupation"].str.endswith("_teacher")]
    assert (educators["age"] >= 15).all()
    assert (educators["age"] >= 18).mean() > 0.95
    assert (educators["age"] < 70).all()

    # source: https://tinyurl.com/y3psel4p
    pct_teachers = 782_613 / POPULATION_GERMANY
    assert np.abs((df["occupation"] == "school_teacher").mean() - pct_teachers) < 0.004

    # source: https://tinyurl.com/y2v8zlgo
    pct_preschool_teachers = 380_000 / POPULATION_GERMANY
    share_preschool_teachers = (df["occupation"] == "preschool_teacher").mean()
    assert np.abs(share_preschool_teachers - pct_preschool_teachers) < 0.002


def _check_educ_group_sizes(df):
    name_to_class_bounds = {
        # school target is 23 pupils + 2 teachers => 25 +/- 5
        "school": (20, 30, 2),
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
        "school_group_id_0": (5, [2, 3]),
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
