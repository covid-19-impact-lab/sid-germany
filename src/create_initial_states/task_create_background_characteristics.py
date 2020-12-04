"""Create a synthetic population that is representative of Germany."""
import numpy as np
import pytask

from src.config import BLD
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
    workers = df.query("occupation == 'working'")
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
    assert not df[df["occupation"] != "working"]["systemically_relevant"].any()
    assert workers["systemically_relevant"].mean() > 0.27
    assert workers["systemically_relevant"].mean() < 0.35


def _check_group_ids(
    df,
    work_daily_dist,
    work_weekly_dist,
    other_daily_dist,
    other_weekly_dist,
):
    df = df.copy(deep=True)

    # create helpers
    w_weekly_cols = [x for x in df if x.startswith("work_weekly_group")]
    o_weekly_cols = [x for x in df if x.startswith("other_weekly_group")]
    n_weekly_w_groups = df[w_weekly_cols].replace(-1, np.nan).notnull().sum(axis=1)
    df["n_weekly_w_groups"] = n_weekly_w_groups
    n_weekly_o_groups = df[o_weekly_cols].replace(-1, np.nan).notnull().sum(axis=1)
    df["n_weekly_o_groups"] = n_weekly_o_groups

    workers = df.query("occupation == 'working'")
    non_workers = df.query("occupation != 'working'")

    # weekly group ids
    assert len(w_weekly_cols) == 14
    assert len(o_weekly_cols) == 8
    assert (non_workers[w_weekly_cols] == -1).all().all()
    w_weekly_size_shares = workers["n_weekly_w_groups"].value_counts(normalize=True)
    o_weekly_size_shares = df["n_weekly_o_groups"].value_counts(normalize=True)
    assert np.abs(w_weekly_size_shares - work_weekly_dist).max() < 0.04
    assert np.abs(o_weekly_size_shares - other_weekly_dist).max() < 0.08

    # daily work group ids
    w_daily_group_vc = workers["work_daily_group_id"].value_counts()
    # drop -1 category
    w_daily_group_vc = w_daily_group_vc[w_daily_group_vc > 0]
    assert w_daily_group_vc.max() <= 16
    assert (non_workers["work_daily_group_id"] == -1).all()
    assert (workers["work_daily_group_id"] != -1).all()
    # compare true and target distribution (incomplete!)
    w_daily_group_size_shares = w_daily_group_vc.value_counts(normalize=True)
    assert w_daily_group_size_shares[::-1].is_monotonic
    goal_w_daily_group_size_shares = work_daily_dist.copy(deep=True)
    goal_w_daily_group_size_shares.index += 1
    assert w_daily_group_size_shares.argmax() == goal_w_daily_group_size_shares.argmax()

    # daily other group ids
    o_daily_group_vc = df["other_daily_group_id"].value_counts()
    # drop -1 category
    o_daily_group_vc = o_daily_group_vc[o_daily_group_vc > 0]
    assert o_daily_group_vc.max() <= 6
    o_daily_group_size_shares = o_daily_group_vc.value_counts(normalize=True)
    goal_o_daily_group_size_shares = other_daily_dist.copy(deep=True)
    goal_o_daily_group_size_shares.index += 1
    diff_btw_o_shares = o_daily_group_size_shares - goal_o_daily_group_size_shares
    assert np.abs(diff_btw_o_shares).max() < 0.1
