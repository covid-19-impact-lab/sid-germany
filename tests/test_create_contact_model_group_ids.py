import pandas as pd

from src.create_initial_states.create_contact_model_group_ids import (
    _sample_household_groups,
)


def test_sample_household_groups():
    df = pd.DataFrame()
    df["hh_id"] = [0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7]
    df["private_hh"] = True

    # this is a regression test
    res = _sample_household_groups(df, 333)
    expected = pd.Series([0] * 5 + [1] * 3 + [2] * 3 + [1] * 5 + [2])
    assert (res == expected).all()


def test_sample_household_groups_with_non_private_hh():
    df = pd.DataFrame()
    df["hh_id"] = [0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 8, 8, 9, 9]
    df["private_hh"] = [True] * 16 + [False] + [True] * 4

    df["res"] = _sample_household_groups(df, 333)
    assert df["res"][16] == -1
    matched = df.query("private_hh")
    assert (matched.groupby("hh_id")["res"].nunique() == 1).all()
    # -1 still occurs as category with 0 values
    assert (matched.groupby("res")["hh_id"].nunique().isin([0, 3])).all()
