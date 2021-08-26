import warnings

import pandas as pd
import pytest

from src.policies.find_people_to_vaccinate import find_people_to_vaccinate


@pytest.fixture
def params():
    index = pd.MultiIndex.from_tuples(
        [
            ("vaccinations", "share_refuser", "adult"),
            ("vaccinations", "share_refuser", "youth"),
        ]
    )
    params = pd.DataFrame(
        data=0.0,
        index=index,
        columns=["value"],
    )
    return params


def test_find_people_to_vaccinate_no_refusers(params):
    states = pd.DataFrame()
    states["vaccination_rank"] = [0.35, 0.45, 0.25, 0.15, 0.85, 0.55]
    states["date"] = pd.Timestamp("2021-02-03")

    vaccination_shares = pd.Series(
        [0.1, 0.2, 0.1],  # 0.3 to 0.4 should get vaccinated
        index=[
            pd.Timestamp("2021-02-01"),
            pd.Timestamp("2021-02-02"),
            pd.Timestamp("2021-02-03"),
        ],
    )
    # 0.15 and 0.25 are already vaccinated
    # 0.35 is vaccinated this period. 0.45 and above not yet.
    expected = pd.Series([True] + [False] * 5)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        res = find_people_to_vaccinate(
            receives_vaccine=None,
            states=states,
            params=params,
            seed=33,
            vaccination_shares=vaccination_shares,
            init_start=pd.Timestamp("2021-01-15"),
        )

    pd.testing.assert_series_equal(expected, res, check_names=False)


def test_find_people_to_vaccinate_with_refusers(params):
    states = pd.DataFrame()
    states["vaccination_rank"] = [0.35, 0.45, 0.25, 0.15, 0.85, 0.55]
    states["date"] = pd.Timestamp("2021-02-03")

    params["value"] = [0.3, 0.0]

    vaccination_shares = pd.Series(
        # 0.3 to 0.9 should get vaccinated, b/c of refusers only 0.3 to 0.7
        [
            0.1,
            0.2,
            0.6,
        ],
        index=[
            pd.Timestamp("2021-02-01"),
            pd.Timestamp("2021-02-02"),
            pd.Timestamp("2021-02-03"),
        ],
    )
    # 0.15 and 0.25 are already vaccinated
    # 0.35, 0.45, 0.55 get vaccinated. 0.85 belongs to the refusers.
    expected = pd.Series([True, True, False, False, False, True])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        res = find_people_to_vaccinate(
            receives_vaccine=None,
            states=states,
            params=params,
            seed=33,
            vaccination_shares=vaccination_shares,
            init_start=pd.Timestamp("2021-01-15"),
        )

    pd.testing.assert_series_equal(expected, res, check_names=False)


def test_find_people_to_vaccinate_start_date(params):
    states = pd.DataFrame()
    states["vaccination_rank"] = [0.35, 0.45, 0.25, 0.15, 0.85, 0.55]
    states["date"] = pd.Timestamp("2021-02-03")

    vaccination_shares = pd.Series(
        [0.1, 0.2, 0.1],
        index=[
            pd.Timestamp("2021-02-01"),
            pd.Timestamp("2021-02-02"),
            pd.Timestamp("2021-02-03"),
        ],
    )
    # everyone up to 0.4 should be vaccinated
    expected = pd.Series([True, False, True, True, False, False])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        res = find_people_to_vaccinate(
            receives_vaccine=None,
            states=states,
            params=params,
            seed=33,
            vaccination_shares=vaccination_shares,
            init_start=pd.Timestamp("2021-02-03"),
        )

    pd.testing.assert_series_equal(expected, res, check_names=False)


def test_find_people_to_vaccinate_after_june_7(params):
    states = pd.DataFrame()
    states["age"] = [30, 40, 50, 14, 10, 15]
    states["vaccination_group_with_refuser_group"] = [1, 6, 4, 5, 5, 5]
    states["ever_vaccinated"] = [True, False, False, False, False, True]
    states["date"] = pd.Timestamp("2021-08-03")

    params["value"] = [0.3, 0.0]

    vaccination_shares = pd.Series(
        [
            0.1,
            0.2,
            0.35,  # 2 individuals should get vaccinated
        ],
        index=[
            pd.Timestamp("2021-08-01"),
            pd.Timestamp("2021-08-02"),
            pd.Timestamp("2021-08-03"),
        ],
    )
    # 0 and -1 already vaccinated -> must be False
    # -2 is too young -> must be False
    # 1 is a refuser -> must be False
    expected = pd.Series([False, False, True, True, False, False])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        res = find_people_to_vaccinate(
            receives_vaccine=None,
            states=states,
            params=params,
            seed=33,  # result is dependent on the seed!
            vaccination_shares=vaccination_shares,
            init_start=pd.Timestamp("2021-01-15"),
        )

    pd.testing.assert_series_equal(expected, res, check_names=False)
