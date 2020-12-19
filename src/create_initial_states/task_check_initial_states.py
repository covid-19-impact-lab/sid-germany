import numpy as np
import pandas as pd
import pytask
from pandas.api.types import is_categorical_dtype

from src.config import BLD
from src.config import POPULATION_GERMANY


@pytask.mark.depends_on(
    {
        "initial_states": BLD / "data" / "initial_states.parquet",
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
def task_check_initial_states(depends_on):
    df = pd.read_parquet(depends_on["initial_states"])
    _check_background_characteristics(df)

    work_daily_dist = pd.read_pickle(depends_on["work_daily_dist"])
    work_weekly_dist = pd.read_pickle(depends_on["work_weekly_dist"])
    other_daily_dist = pd.read_pickle(depends_on["other_daily_dist"])
    other_weekly_dist = pd.read_pickle(depends_on["other_weekly_dist"])

    _check_systemically_relevant(df)
    _check_work_contact_priority(df)
    _check_educ_group_ids(df)
    _check_work_group_ids(df, work_daily_dist, work_weekly_dist)
    _check_other_group_ids(df, other_daily_dist, other_weekly_dist)
    for i in range(3):
        col = f"christmas_group_id_{i}"
        _check_christmas_groups(df, col)
    assert (df["christmas_group_id_0"] != df["christmas_group_id_1"]).any()
    not_categorical_group_ids = [
        col for col in df if "group_id" in col and not is_categorical_dtype(df[col])
    ]
    assert (
        len(not_categorical_group_ids) == 0
    ), f"There are non categorical group id columns: {not_categorical_group_ids}"


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
    old_women_share = (df.query("70 < age < 80")["gender"] == "female").mean()
    assert old_women_share > 0.53
    old_women_share = (df.query("age > 80")["gender"] == "female").mean()
    assert old_women_share > 0.62
    assert (df.query("occupation == 'working'")["gender"] == "female").mean() < 0.5
    # given couple formation men and women appear in much more than half of households
    assert df.groupby("hh_id")["female"].any().mean() > 0.75
    assert df.groupby("hh_id")["male"].any().mean() > 0.75

    _check_occupation_column(df)


def _check_occupation_column(df):
    df = df.copy(deep=True)
    educ_worker_categories = [
        "school_teacher",
        "preschool_teacher",
        "nursery_teacher",
    ]
    df["occupation"] = df["occupation"].replace(
        {cat: "working" for cat in educ_worker_categories}
    )

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

    assert set(df["school_group_a"].unique()) == {1, 0}
    assert df.query("age < 3")["occupation"].isin(["nursery", "stays home"]).all()
    assert (df.query("3 <= age <= 14")["nursery_group_id_0"] == -1).all()
    preschool_kid_groups = df.query("3 <= age < 6")["preschool_group_id_0"]
    assert (preschool_kid_groups != -1).all()
    assert (preschool_kid_groups.value_counts().drop(-1).isin([8, 9, 10])).all()
    kids = df.query("6 < age <= 14")
    assert (kids["preschool_group_id_0"] == -1).all()
    assert (kids["school_group_id_0"] != -1).all()
    assert (kids["school_group_id_0"].astype(int).value_counts() > 1).all()

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
    df = df.copy(deep=True)
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
        df[id_col] = df[id_col].astype(float)

        pupils_and_teachers = df[df[id_col] != -1]
        class_sizes = pupils_and_teachers[id_col].value_counts().unique()
        assert (class_sizes >= lower).all()
        assert (class_sizes <= upper).all()
        n_teachers = pupils_and_teachers.groupby(id_col)["occupation"].apply(
            lambda x: (x == f"{name}_teacher").sum()
        )
        assert (n_teachers == expected_n_teachers).all()


def _check_educ_group_assortativeness(df):
    df = df.copy(deep=True)
    col_to_limits = {
        "nursery_group_id_0": (3, [2, 3, 4]),
        "preschool_group_id_0": (4, [3, 4, 5]),
        "school_group_id_0": (7, [2, 3]),
    }
    for col, (max_counties, allowed_n_ages) in col_to_limits.items():
        df[col] = df[col].astype(int)
        pupils_and_teachers = df[df[col] != -1]
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


def _check_christmas_groups(df, col):
    assert df[col].notnull().all(), f"No NaNs allowed in {col}."
    community_groups = df.query("~private_hh")[col]
    assert (community_groups == -1).all(), "Only private hhs should be matched"

    df = df.query("private_hh").copy(deep=True)
    df["hh_id"] = df["hh_id"].astype(float)
    groups_per_hh = df.groupby("hh_id")[col].nunique()
    assert (groups_per_hh == 1).all(), "Every hh must have one christmas group."
    hh_per_group = df.groupby(col)["hh_id"].nunique()
    assert hh_per_group[-1] == 0
    assert (
        hh_per_group.drop(-1) == 3
    ).mean() > 0.9999, "Too many groups don't have 3 households."