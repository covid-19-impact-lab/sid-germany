import numpy as np
import pandas as pd

from src.create_initial_states.make_educ_group_columns import (
    _create_group_id_for_non_participants,
)
from src.create_initial_states.make_educ_group_columns import (
    _create_group_id_for_one_strict_assort_by_group,
)
from src.create_initial_states.make_educ_group_columns import (
    _create_group_id_for_participants,
)
from src.create_initial_states.make_educ_group_columns import _determine_group_sizes
from src.create_initial_states.make_educ_group_columns import _get_id_to_weak_group
from src.create_initial_states.make_educ_group_columns import (
    _get_key_with_longest_value,
)
from src.create_initial_states.make_educ_group_columns import (
    _get_key_with_shortest_value,
)
from src.create_initial_states.make_educ_group_columns import _split_data_by_query


def test_get_id_to_weak_group():
    raw_id = pd.Series([2, 2, 3, 3, 4, 4, 5, 5])  # dtype int is right.
    participants = pd.DataFrame(index=[2, 3, 4, 5])
    participants["__weak_group_id"] = [0, 1] + [1, 0]
    expected = pd.Series([0, 1], index=[3, 4])
    res = _get_id_to_weak_group(participants, raw_id)
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_split_data_by_query():
    df = pd.DataFrame(index=list("abcde"))
    df["to_select"] = [True, True, False, True, False]
    query = "to_select"
    res_selected, res_others = _split_data_by_query(df, query)
    expected_selected = df.loc[["a", "b", "d"]]
    expected_other = df.loc[["c", "e"]]
    pd.testing.assert_frame_equal(res_selected, expected_selected)
    pd.testing.assert_frame_equal(res_others, expected_other)


def test_create_group_id_for_participants():
    df = pd.DataFrame()
    df["state"] = ["BY"] * 4 + ["NRW"] * 8
    df["county"] = ["N", "N", "M", "M"] + ["K"] * 5 + ["D"] * 3
    group_size = 2
    strict_assort_by = "state"
    weak_assort_by = "county"
    res = _create_group_id_for_participants(
        df=df,
        group_size=group_size,
        strict_assort_by=strict_assort_by,
        weak_assort_by=weak_assort_by,
    )
    expected = pd.Series(
        [2, 2, 1, 1, 4, 4, 6, 6, 7, 5, 5, 7], dtype=float, name="group_id"
    )
    pd.testing.assert_series_equal(res, expected)


def test_create_group_id_for_one_strict_assort_by_group_one_county_size_one():
    df = pd.DataFrame()
    df["weak_assort_by"] = ["a", "a", "a", "a"]
    group_size = 1
    weak_assort_by = "weak_assort_by"
    start_id = 20
    res, end_id = _create_group_id_for_one_strict_assort_by_group(
        df=df, group_size=group_size, weak_assort_by=weak_assort_by, start_id=start_id
    )
    expected = pd.Series([20.0, 21.0, 22.0, 23.0], index=df.index, name="group_id")
    pd.testing.assert_series_equal(expected, res)
    assert end_id == 24


def test_create_group_id_for_one_strict_assort_by_group_no_remainder():
    df = pd.DataFrame()
    df["weak_assort_by"] = ["a", "b", "a", "b"]
    group_size = 2
    weak_assort_by = "weak_assort_by"
    start_id = 20
    res, end_id = _create_group_id_for_one_strict_assort_by_group(
        df=df, group_size=group_size, weak_assort_by=weak_assort_by, start_id=start_id
    )
    expected = pd.Series([21.0, 20.0, 21.0, 20.0], index=df.index, name="group_id")
    pd.testing.assert_series_equal(expected, res)
    assert end_id == 22


def test_create_group_id_for_one_strict_assort_by_group_with_remainder():
    df = pd.DataFrame()
    df["weak_assort_by"] = ["a", "b", "a", "b", "a", "a", "a"]
    group_size = 2
    weak_assort_by = "weak_assort_by"
    start_id = 20
    res, end_id = _create_group_id_for_one_strict_assort_by_group(
        df=df, group_size=group_size, weak_assort_by=weak_assort_by, start_id=start_id
    )
    expected = pd.Series(
        [20.0, 22.0, 20.0, 22.0, 21.0, 21.0, 23.0], index=df.index, name="group_id"
    )
    pd.testing.assert_series_equal(expected, res)
    assert end_id == 24


def test_determine_group_sizes_target_one():
    res = _determine_group_sizes(target_size=1, population_size=5)
    expected = [1, 1, 1, 1, 1]
    assert res == expected


def test_determine_group_sizes_no_remainder():
    res = _determine_group_sizes(target_size=2, population_size=6)
    expected = [2, 2, 2]
    assert res == expected


def test_determine_group_sizes_remainder():
    res = _determine_group_sizes(target_size=3, population_size=11)
    expected = [3, 3, 3, 2]
    assert res == expected


def test_get_key_with_shortest_value():
    d = {"a": "a", "b": "uiae", "c": np.array([1, 2, 3])}
    expected = "a"
    res = _get_key_with_shortest_value(d)
    assert res == expected


def test_get_key_with_shortest_value_non_unique_solution():
    d = {"a": "a", "b": "b", "c": np.array([1, 2, 3])}
    expected = "a"
    res = _get_key_with_shortest_value(d)
    assert res == expected


def test_get_key_with_longest_value():
    d = {"a": "a", "b": "uiae", "c": np.array([1, 2, 3])}
    expected = "b"
    res = _get_key_with_longest_value(d)
    assert res == expected


def test_get_key_with_longest_value_non_unique_solution():
    d = {"a": "abcd", "b": "abcd", "c": "c"}
    expected = "b"
    res = _get_key_with_longest_value(d)
    assert res == expected


def test_create_group_id_for_non_participants():
    df = pd.DataFrame(index=[2, 3, 4])
    res = _create_group_id_for_non_participants(df, 20)
    expected = pd.Series([20, 21, 22], index=[2, 3, 4])
    pd.testing.assert_series_equal(res, expected)
