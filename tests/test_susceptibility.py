import numpy as np
import pandas as pd
from src.simulation.calculate_susceptibility import calculate_susceptibility
from src.config import SRC
import pytest


@pytest.fixture
def params():
    params = pd.DataFrame()
    params["name"] = [
        "0-9",
        "10-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70-79",
        "80-100",
    ]
    params["value"] = np.arange(0.0, 0.9, 0.1)
    params["category"] = "susceptibility"
    params["subcategory"] = "susceptibility"
    params = params.set_index(["category", "subcategory", "name"])
    return params


def test_calculate_susceptiblitiy(params):
    # every age group appears once
    states = pd.read_pickle(SRC.parent / "tests" / "age_groups.pkl")

    res = calculate_susceptibility(states=states, params=params)
    expected = pd.Series(
        [0.5, 0.3, 0.2, 0.1, 0.6, 0.7, 0.0, 0.8, 0.4],
        index=states.index,
        name="susceptibility",
    )
    pd.testing.assert_series_equal(res, expected)
