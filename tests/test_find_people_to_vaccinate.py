import pandas as pd

from src.policies.find_people_to_vaccinate import find_people_to_vaccinate


def test_find_people_to_vaccinate():
    states = pd.DataFrame()
    states["vaccination_rank"] = []
    states["date"] = pd.Timestamp("2021-02-02")

    params = None
    seed = 384
    vaccination_shares = pd.Series(
        [0.1, 0.2, 0.1],
        index=[
            pd.Timestamp("2021-02-01"),
            pd.Timestamp("2021-02-02"),
            pd.Timestamp("2021-02-03"),
        ],
    )
