import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from src.shared import _create_group_ids
from src.shared import _determine_number_of_groups
from src.shared import _expand_or_contract_ids
from src.shared import create_groups_from_dist
from src.shared import draw_groups


@pytest.fixture
def df():
    df = pd.DataFrame(
        data={
            "age": [15, 25, 30, 70, 20, 25],
            "region": ["A", "B", "B", "B", "A", "A"],
        },
        columns=["age", "region"],
    )
    return df


def test_draw_groups(df):
    res = draw_groups(
        df=df,
        query="18 <= age <= 65",
        assort_bys=["region"],
        n_per_group=20,
        seed=393,
    )
    expected = np.array([-1, 1, 1, -1, 0, 0])
    assert_array_equal(res.to_numpy(), expected)


def test_determine_number_of_groups():
    nobs = 40
    dist = pd.Series({1: 0.5, 2: 0.25, 5: 0.25})
    expected = pd.Series({1: 20, 2: 5, 5: 2})
    res = _determine_number_of_groups(nobs=nobs, dist=dist)
    assert_series_equal(res, expected)


def test_create_group_ids():
    assort_by_vals = ("hello", "world")
    nr_of_groups = pd.Series({1: 20, 2: 5, 5: 2})
    prefix = f"{assort_by_vals}_"
    without_prefix_expected = (
        [f"1_{i}" for i in range(20)]
        + ["2_0", "2_0", "2_1", "2_1", "2_2", "2_2", "2_3", "2_3", "2_4", "2_4"]
        + ["5_0", "5_0", "5_0", "5_0", "5_0", "5_1", "5_1", "5_1", "5_1", "5_1"]
    )
    expected = [prefix + s for s in without_prefix_expected]
    res = _create_group_ids(nr_of_groups, assort_by_vals)
    assert res == expected


def test_expand_or_contract_ids_add_two():
    ids = [0, 2, 4, 5, 7, 9]
    nobs = 8
    expected = np.array([0, 2, 4, 5, 7, 9, "bla_2_rest", "bla_2_rest"])
    res = _expand_or_contract_ids(ids, nobs, "bla")
    assert (res == expected).all()


def test_expand_or_contract_ids_need_remove_one():
    ids = [0, 2, 4, 5, 7, 9]
    nobs = 5
    expected = np.array([2, 4, 5, 7, 9])
    res = _expand_or_contract_ids(ids, nobs, "foo")
    assert (res == expected).all()


def test_create_groups_from_dist(monkeypatch):
    monkeypatch.setattr(np.random, "choice", lambda x, size, replace: x)

    assort_bys = ["assort1"]
    query = "a == 1"
    initial_states = pd.DataFrame()
    initial_states["a"] = [1] * 24 + [0]
    initial_states["assort1"] = [0] * 12 + [1] * 13

    group_distribution = pd.Series([0.5, 0.5], index=[3, 6], name="group_sizes")
    res = create_groups_from_dist(
        initial_states=initial_states,
        group_distribution=group_distribution,
        query=query,
        assort_bys=assort_bys,
        seed=3944,
    )

    expected = pd.Series(
        ["0_3_0", "0_3_0", "0_3_0", "0_3_1", "0_3_1", "0_3_1"]
        + ["0_6_0"] * 6
        + ["1_3_0", "1_3_0", "1_3_0", "1_3_1", "1_3_1", "1_3_1"]
        + ["1_6_0"] * 6
        + [-1],
        index=initial_states.index,
        dtype="category",
    )
    assert_series_equal(res, expected)
