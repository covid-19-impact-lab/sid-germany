import pandas as pd
import pytest

from src.simulation.params_scenarios import _build_new_date_params


@pytest.fixture
def params():
    params = pd.DataFrame()
    params["value"] = [0.3, 0.6, 0.9]
    params["name"] = ["2021-01-01", "2021-04-01", "2021-04-30"]
    params["subcategory"] = "subcategory"
    params["category"] = "category"
    params = params.set_index(["category", "subcategory", "name"])
    params.loc[("other", "other", "other")] = 15
    return params


def test_build_new_date_params(params):
    res = _build_new_date_params(
        params.loc[("category", "subcategory")],
        change_date=pd.Timestamp("2021-04-17"),
        new_val=1.0,
    )
    expected_index = pd.DatetimeIndex(
        ["2021-01-01", "2021-04-01", "2021-04-16", "2021-04-17", "2025-12-31"],
        name="name",
    )
    expected = pd.DataFrame(index=expected_index)
    expected["value"] = [
        0.3,  # kept
        0.6,  # kept
        0.7551724137931035,  # interpolated value right before the change
        1.0,  # on the change date
        1.0,  # maintain value
    ]

    pd.testing.assert_frame_equal(res, expected)
