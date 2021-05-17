import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.create_initial_states.create_initial_conditions import (
    _create_group_specific_share_known_cases,
)
from src.create_initial_states.create_initial_conditions import (
    _scale_up_empirical_new_infections,
)
from src.create_initial_states.create_initial_infections import (
    _add_variant_info_to_infections,
)
from src.create_initial_states.create_initial_infections import (
    _calculate_group_infection_probs,
)


@pytest.fixture
def empirical_infections():
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
        cases=undetected_multiplier * cases,
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


def test_add_variant_info_to_infections():
    df = pd.DataFrame()
    dates = [pd.Timestamp("2021-03-14"), pd.Timestamp("2021-03-15")]
    df[dates[0]] = [False, True] * 5
    df[dates[1]] = [False] * 8 + [True, False]
    virus_shares = {
        "base_strain": pd.Series([1, 0.5], index=dates),
        "other_strain": pd.Series([0, 0.5], index=dates),
    }
    np.random.seed(39223)
    expected = pd.DataFrame()
    expected[dates[0]] = pd.Categorical(
        [np.nan, "base_strain"] * 5, categories=["base_strain", "other_strain"]
    )
    expected[dates[1]] = pd.Categorical(
        [np.nan] * 8 + ["other_strain", np.nan],
        categories=["base_strain", "other_strain"],
    )
    res = _add_variant_info_to_infections(bool_df=df, virus_shares=virus_shares)
    pdt.assert_frame_equal(res, expected)


GROUPS = ["0-4", "5-14", "15-34", "35-59", "60-79", "80-100"]

DATES = pd.date_range("2021-04-01", "2021-04-03")


@pytest.fixture
def overall_share_known_cases():
    sr = pd.Series([0.3, 0.5, 0.7], index=DATES)
    sr.index.name = "date"
    return sr


@pytest.fixture
def group_weights():
    sr = pd.Series([0.1, 0.1, 0.2, 0.3, 0.2, 0.1], index=GROUPS)
    sr.index.name = "age_group_rki"
    return sr


@pytest.fixture
def group_share_known_cases():
    sr = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], index=GROUPS)
    sr.index.name = "age_group_rki"
    return sr


def test_create_group_specific_share_known_cases_just_overall(
    overall_share_known_cases, group_weights
):
    res = _create_group_specific_share_known_cases(
        overall_share_known_cases=overall_share_known_cases,
        group_share_known_cases=None,
        group_weights=group_weights,
        date_range=DATES,
    )
    expected = pd.DataFrame(
        [[0.3] * 6, [0.5] * 6, [0.7] * 6], index=DATES, columns=GROUPS
    )

    assert res.equals(expected)


def test_create_group_specific_share_known_cases_just_group(
    group_share_known_cases, group_weights
):
    res = _create_group_specific_share_known_cases(
        overall_share_known_cases=None,
        group_share_known_cases=group_share_known_cases,
        group_weights=group_weights,
        date_range=DATES,
    )
    expected = pd.DataFrame(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        index=DATES,
        columns=GROUPS,
    )
    assert res.equals(expected)


def test_create_group_specific_share_known_cases_both_given(
    overall_share_known_cases,
    group_weights,
    group_share_known_cases,
):

    res = _create_group_specific_share_known_cases(
        overall_share_known_cases=overall_share_known_cases,
        group_share_known_cases=group_share_known_cases,
        group_weights=group_weights,
        date_range=DATES,
    )

    # implied overall: 0.01 + 0.02 + 0.06 + 0.12 + 0.1 + 0.06 = 0.37
    # Some
    expected = pd.DataFrame(
        [
            # 0-4         5-14          14-34       35-59       60-79       80-100
            [0.08108108, 0.16216216, 0.24324324, 0.32432432, 0.40540541, 0.48648649],
            [0.13513514, 0.27027027, 0.40540541, 0.54054054, 0.67567568, 0.81081081],
            [0.18918919, 0.37837838, 0.56756757, 0.75675676, 0.94594595, 1.13513514],
        ],
        index=DATES,
        columns=GROUPS,
    )
    expected.columns.name = "age_group_rki"
    expected.index.name = "date"

    pd.testing.assert_series_equal(res @ group_weights, overall_share_known_cases)
    pd.testing.assert_frame_equal(res, expected)


@pytest.fixture
def empirical_infections_for_upscaling():
    df = pd.DataFrame()
    df["date"] = [date for date in DATES for _ in GROUPS]
    df["age_group_rki"] = [g for g in GROUPS for _ in DATES]
    df["county"] = ["A", "B"] * 9
    df["newly_infected"] = [10, 1, 2, 3] * 4 + [10, 10]
    df = df.set_index(["date", "county", "age_group_rki"])
    return df


def test_scale_up_empirical_new_infections(
    empirical_infections_for_upscaling,
    group_share_known_cases,
    group_weights,
    overall_share_known_cases,
):
    empirical_infections = empirical_infections_for_upscaling.loc[:"2021-04-02"]
    res = _scale_up_empirical_new_infections(
        empirical_infections=empirical_infections,
        group_share_known_cases=group_share_known_cases,
        group_weights=group_weights,
        overall_share_known_cases=overall_share_known_cases,
    )
    expected = empirical_infections.copy(deep=True)
    expected["upscaled_newly_infected"] = [
        # 2021-04-01
        # 0-4
        10 / 0.08108108,
        1 / 0.08108108,
        2 / 0.08108108,
        # 5-14
        3 / 0.16216216,
        10 / 0.16216216,
        1 / 0.16216216,
        # 2021-04-02
        # 15-34
        2 / 0.40540541,
        3 / 0.40540541,
        10 / 0.40540541,
        # 35-59
        1 / 0.54054054,
        2 / 0.54054054,
        3 / 0.54054054,
    ]
    pd.testing.assert_series_equal(res, expected["upscaled_newly_infected"])


def test_scale_up_empirical_new_infections_raise_error(
    empirical_infections_for_upscaling,
    group_share_known_cases,
    group_weights,
    overall_share_known_cases,
):
    msg = "The group specific share known cases is > 1 for some date and group."
    with pytest.raises(AssertionError) as excinfo:
        _scale_up_empirical_new_infections(
            empirical_infections=empirical_infections_for_upscaling,
            group_share_known_cases=group_share_known_cases,
            group_weights=group_weights,
            overall_share_known_cases=overall_share_known_cases,
        )
        assert msg in str(excinfo.value)
