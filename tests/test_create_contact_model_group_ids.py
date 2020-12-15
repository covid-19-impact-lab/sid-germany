import pandas as pd

from src.create_initial_states.create_contact_model_group_ids import _draw_other_group
from src.create_initial_states.create_contact_model_group_ids import (
    _sample_household_groups,
)


def test_sample_household_groups_no_assort():
    df = pd.DataFrame()
    df["hh_id"] = [0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7]
    df["private_hh"] = True
    df["res"] = _sample_household_groups(df, seed=333, assort_by=None)
    assert (df.groupby("hh_id")["res"].nunique() == 1).all()
    assert (df.groupby("res")["hh_id"].nunique().isin([2, 3])).all()


def test_sample_household_groups_with_non_private_hh():
    df = pd.DataFrame()
    df["hh_id"] = [0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 8, 8, 9, 9]
    df["private_hh"] = [True] * 16 + [False] + [True] * 4

    df["res"] = _sample_household_groups(df, 333, assort_by=None)
    assert df["res"][16] == -1
    matched = df.query("private_hh")
    assert (matched.groupby("hh_id")["res"].nunique() == 1).all()
    # -1 still occurs as category with 0 values
    assert (matched.groupby("res")["hh_id"].nunique().isin([0, 3])).all()


def test_sample_household_groups_with_assort_by():
    df = pd.DataFrame()
    df["hh_id"] = [0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 8, 8]
    df["private_hh"] = [True] * 16 + [False] + [True] * 2

    df["state"] = df["hh_id"].apply(lambda x: "A" if x <= 5 else "B")
    df["hh_id"] = df["hh_id"].astype("category")

    df["res"] = _sample_household_groups(
        df, seed=334, assort_by="state", same_group_probability=1.0
    )
    assert (df.groupby("res")["state"].nunique() == 1).all()
    assert (df.groupby("hh_id")["res"].nunique() == 1).all()


def test_draw_other_group_empty_group():
    grouped_to_match = {"a": [], "b": [], "c": [2]}
    group = "a"
    same_group_probability = 1.0
    for i in range(10):
        res = _draw_other_group(grouped_to_match, group, same_group_probability, i)
        assert res == "c"


def test_draw_other_group_same_group():
    grouped_to_match = {"a": [3], "b": [], "c": [2]}
    group = "a"
    same_group_probability = 1.0
    for i in range(10):
        res = _draw_other_group(grouped_to_match, group, same_group_probability, i)
        assert res == "a"


def test_draw_other_group_one_other_group():
    grouped_to_match = {"a": [3], "b": [], "c": [2]}
    group = "a"
    same_group_probability = 0.0
    for i in range(10):
        res = _draw_other_group(grouped_to_match, group, same_group_probability, i)
        assert res == "c"


def test_draw_other_group_two_other_groups():
    grouped_to_match = {"a": [3], "b": [1], "c": [2]}
    group = "a"
    same_group_probability = 0.0
    for i in range(10):
        res = _draw_other_group(grouped_to_match, group, same_group_probability, i)
        assert res in ["b", "c"]
