import pandas as pd

from src.policies.find_people_to_vaccinate import find_people_to_vaccinate


def test_find_people_to_vaccinate_no_refusers():
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

    res = find_people_to_vaccinate(
        receives_vaccine=None,
        states=states,
        params=None,
        seed=33,
        vaccination_shares=vaccination_shares,
        no_vaccination_share=1.0,
        init_start=pd.Timestamp("2021-01-15"),
    )

    pd.testing.assert_series_equal(expected, res, check_names=False)


def test_find_people_to_vaccinate_with_refusers():
    states = pd.DataFrame()
    states["vaccination_rank"] = [0.35, 0.45, 0.25, 0.15, 0.85, 0.55]
    states["date"] = pd.Timestamp("2021-02-03")

    vaccination_shares = pd.Series(
        [0.1, 0.2, 0.5],  # 0.3 to 0.8 should get vaccinated
        index=[
            pd.Timestamp("2021-02-01"),
            pd.Timestamp("2021-02-02"),
            pd.Timestamp("2021-02-03"),
        ],
    )
    # 0.15 and 0.25 are already vaccinated
    # 0.35 and 0.45 get vaccinated. 0.55 belongs to the refusers.
    expected = pd.Series([True, True] + [False] * 4)

    res = find_people_to_vaccinate(
        receives_vaccine=None,
        states=states,
        params=None,
        seed=33,
        vaccination_shares=vaccination_shares,
        no_vaccination_share=0.5,
        init_start=pd.Timestamp("2021-01-15"),
    )

    pd.testing.assert_series_equal(expected, res, check_names=False)


def test_find_people_to_vaccinate_start_date():
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

    res = find_people_to_vaccinate(
        receives_vaccine=None,
        states=states,
        params=None,
        seed=33,
        vaccination_shares=vaccination_shares,
        no_vaccination_share=1.0,
        init_start=pd.Timestamp("2021-02-03"),
    )

    pd.testing.assert_series_equal(expected, res, check_names=False)
