import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.create_initial_states.create_initial_infections import _create_cases
from src.create_initial_states.create_initial_infections import _only_leave_first_true


def test_only_leave_first_true():
    df = pd.DataFrame(
        columns=["a", "b", "c"],
        data=[
            [True, True, False],
            [False, True, True],
            [False, False, True],
            [False, False, False],
            [True, True, True],
        ],
    )
    expected = pd.DataFrame(
        columns=["a", "b", "c"],
        data=[
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, False],
            [True, False, False],
        ],
    )
    res = _only_leave_first_true(df)
    pdt.assert_frame_equal(res, expected)


@pytest.fixture
def empirical_data():
    start = pd.Timestamp("2020-09-30")
    a_day = pd.Timedelta(days=1)
    df = pd.DataFrame()
    df["date"] = [start + i * a_day for i in range(5)] * 4
    df = df.sort_values("date")
    df.reset_index(drop=True, inplace=True)
    df["county"] = list("AABB") * 5
    df["age_group_rki"] = ["young", "old"] * 10
    np.random.seed(3984)
    df["newly_infected"] = np.random.choice([0, 1], 20)
    return df


@pytest.fixture
def cases():
    ind_tuples = [("A", "young"), ("A", "old"), ("B", "young"), ("B", "old")]
    index = pd.MultiIndex.from_tuples(ind_tuples, names=["county", "age_group_rki"])
    df = pd.DataFrame(index=index)
    df["2020-10-01"] = [1, 0, 0, 1]
    df["2020-10-02"] = [1, 1, 0, 0]
    df["2020-10-03"] = [1, 0, 0, 1]
    return df


def test_create_cases(empirical_data, cases):
    start = "2020-10-01"
    end = "2020-10-03"
    res = _create_cases(empirical_data, start, end)
    pdt.assert_frame_equal(res.sort_index(), cases.sort_index())
