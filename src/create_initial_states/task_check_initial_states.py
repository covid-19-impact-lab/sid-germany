import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns

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
        "true_age_group_dist": BLD
        / "data"
        / "population_structure"
        / "age_groups.parquet",
        "vacations": BLD / "data" / "vacations.pkl",
        "work_multiplier": BLD / "policies" / "work_multiplier.csv",
    }
)
@pytask.mark.produces(BLD / "data" / "comparison_of_age_group_distributions.png")
def task_check_initial_states(depends_on, produces):
    df = pd.read_parquet(depends_on["initial_states"])
    true_age_shares = pd.read_parquet(depends_on["true_age_group_dist"])["weight"]
    vacations = pd.read_pickle(depends_on["vacations"])
    work_multiplier = pd.read_csv(depends_on["work_multiplier"])

    _check_federal_states_overlap_btw_initial_states_and_vacation_data(df, vacations)
    _check_background_characteristics(df)
    _check_federal_states_overlap_btw_initial_states_and_work_multiplier(
        df, work_multiplier
    )

    work_daily_dist = pd.read_pickle(depends_on["work_daily_dist"])
    work_weekly_dist = pd.read_pickle(depends_on["work_weekly_dist"])
    other_daily_dist = pd.read_pickle(depends_on["other_daily_dist"])
    other_weekly_dist = pd.read_pickle(depends_on["other_weekly_dist"])

    _check_work_contact_priority(df)
    _check_educ_contact_priority(df)
    _check_educ_group_ids(df)
    _check_work_group_ids(df, work_daily_dist, work_weekly_dist)
    _check_other_group_ids(df, other_daily_dist, other_weekly_dist)

    synthetic_age_shares = df["age_group"].value_counts(normalize=True)
    diff = synthetic_age_shares - true_age_shares
    assert np.abs(diff).max() <= 0.041, (
        "The largest difference between the age group shares in the synthetic "
        "and the true population clearly exceeds 4%."
    )
    assert np.abs(diff).mean() <= 0.015, (
        "The mean difference between the age group shares in the synthetic "
        "and the true population exceeds 1.5%."
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=diff.index, y=diff, color="firebrick", alpha=0.6)
    ax.set_title(
        "Difference between the shares in the initial states and in the "
        "general population\n(> 0 means over represented in the synthetic data)"
    )
    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")
    plt.close()


def _check_federal_states_overlap_btw_initial_states_and_work_multiplier(
    df, work_multiplier
):
    df_states = df["state"].unique()
    work_states = work_multiplier.columns.drop(["date", "Germany"])
    assert set(df_states) == set(work_states), (
        "Federal states don't overlap btw. the initial states and the work "
        "multiplier data."
    )


def _check_federal_states_overlap_btw_initial_states_and_vacation_data(df, vacations):
    df_states = set(df["state"].unique())
    vacc_states = set(vacations.index.get_level_values("subcategory").unique())
    assert (
        df_states == vacc_states
    ), "State names in the vacation data and in the initial states are not the same"


def _check_background_characteristics(df):
    """Check that the background characteristics come out right."""
    df = df.copy(deep=True)

    assert df["hh_id"].value_counts().max() <= 42
    assert df["age"].between(0, 110).all()
    assert (40 <= df["age"].median()) & (48 >= df["age"].median())
    assert df["state"].nunique() == 16
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
    assert df["educ_worker"].notnull().all()

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


def _check_work_contact_priority(df):
    not_working = "occupation in ['stays home', 'retired', 'school']"
    assert (df.query(not_working)["work_contact_priority"] == -1).all()
    workers_priority = df.query("occupation == 'working'")["work_contact_priority"]
    assert workers_priority.between(0.0, 1.0).all()
    assert workers_priority.std() > 0.2
    assert (workers_priority.mean() < 0.52) & (workers_priority.mean() > 0.48)


def _check_educ_contact_priority(df):
    assert df["adult_in_hh_at_home"].notnull().all()
    assert df["educ_contact_priority"].between(0.0, 1.0).all()
    assert (df[df["age"] >= 13]["educ_contact_priority"] == 0).all()
    entitled = df.eval("age < 13 & ~adult_in_hh_at_home")
    max_not_entitled = df[~entitled]["educ_contact_priority"].max()
    min_entitled = df[entitled]["educ_contact_priority"].min()
    share_entitled_children = entitled[df["age"] < 13].mean()
    assert 0.5 < share_entitled_children < 0.6
    assert max_not_entitled <= min_entitled


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

    assert set(df["school_group_id_0_a_b"].unique()) == {1, 0}
    assert df.query("age < 3")["occupation"].isin(["nursery", "stays home"]).all()
    assert (df.query("3 <= age <= 14")["nursery_group_id_0"] == -1).all()
    preschool_kid_groups = df.query("3 <= age < 6")["preschool_group_id_0"]
    assert (preschool_kid_groups != -1).all()
    assert (preschool_kid_groups.value_counts().isin([8, 9, 10])).all()
    kids = df.query("6 < age <= 14")
    assert (kids["preschool_group_id_0"] == -1).all()
    assert (kids["school_group_id_0"] != -1).all()
    assert (kids["school_group_id_0"].astype(int).value_counts() > 1).all()

    _check_educators(df)
    _check_educ_group_sizes(df)
    _check_educ_group_assortativeness(df)

    assert set(df["school_group_id_0_a_b"].unique()) == {0, 1}
    assert 0.49 < df["school_group_id_0_a_b"].mean() < 0.51
    assert (df["school_group_id_0_a_b"] == df["school_group_id_1_a_b"]).all()
    assert (df["school_group_id_0_a_b"] == df["school_group_id_2_a_b"]).all()


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


def _check_group_id_cols_are_factorized(df):
    group_id_cols = [col for col in df if "_group_id" in col]
    for col in group_id_cols:
        unique_non_nan_values = sorted(df[col].unique())
        unique_non_nan_values.remove(-1)
        expected_values = np.arange(len(unique_non_nan_values))
        assert (unique_non_nan_values == expected_values).all()
