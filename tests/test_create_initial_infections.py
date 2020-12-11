import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.create_initial_states.create_initial_infections import (
    _calculate_group_infection_probs,
)


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
    sr = df.set_index(["date", "county", "age_group_rki"])
    return sr


@pytest.fixture
def cases():
    ind_tuples = [("A", "young"), ("A", "old"), ("B", "young"), ("B", "old")]
    index = pd.MultiIndex.from_tuples(ind_tuples, names=["county", "age_group_rki"])
    df = pd.DataFrame(index=index)
    df["2020-10-01"] = [1, 0, 0, 1]
    df["2020-10-02"] = [1, 1, 0, 0]
    df["2020-10-03"] = [1, 0, 0, 1]
    return df


@pytest.fixture
def synthetic_data():
    df = pd.DataFrame()
    df["county"] = list("AABBBBAAA")
    df["age_group_rki"] = ["young"] * 4 + ["old"] * 5
    return df


def test_calculate_group_infection_probs(synthetic_data, cases):
    pop_size = 14
    undetected_multiplier = 1.5
    res = _calculate_group_infection_probs(
        synthetic_data=synthetic_data,
        cases=cases,
        undetected_multiplier=undetected_multiplier,
        population_size=pop_size,
    )
    expected_on_synthetic_data = pd.DataFrame(
        index=synthetic_data.index, columns=cases.columns
    )
    group_shares = np.array([2, 2, 2, 2, 2, 2, 3, 3, 3]) / 9
    scaled_up_group_sizes = pop_size * group_shares
    p1 = (
        undetected_multiplier
        * np.array([1, 1, 0, 0, 1, 1, 0, 0, 0])
        / scaled_up_group_sizes
    )
    p2 = (
        undetected_multiplier
        * np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])
        / scaled_up_group_sizes
    )
    p3 = (
        undetected_multiplier
        * np.array([1, 1, 0, 0, 1, 1, 0, 0, 0])
        / scaled_up_group_sizes
    )

    expected_on_synthetic_data["2020-10-01"] = p1
    expected_on_synthetic_data["2020-10-02"] = p2
    expected_on_synthetic_data["2020-10-03"] = p3

    expected = expected_on_synthetic_data.loc[[0, 2, 4, 6]]
    expected.index = pd.MultiIndex.from_tuples(
        [("A", "young"), ("B", "young"), ("B", "old"), ("A", "old")]
    )
    expected.index.names = ["county", "age_group_rki"]
    pdt.assert_frame_equal(res.sort_index(), expected.sort_index())
