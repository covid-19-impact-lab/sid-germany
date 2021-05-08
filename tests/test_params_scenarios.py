import pandas as pd

from src.simulation.params_scenarios import reduced_rapid_test_demand


def test_reduced_rapid_test_demand():
    params = pd.DataFrame()
    params["subcategory"] = ["share_workers_receiving_offer"] * 3 + [
        "hh_member_demand"
    ] * 3
    params["name"] = ["2021-01-01", "2021-04-01", "2021-0"]
    params["value"] = []
    params["category"] = "rapid_test_demand"
    params = params.set_index(["category", "subcategory", "name"])

    change_date = pd.Timestamp()
    res = reduced_rapid_test_demand(params, change_date, 0.5, 0.2)
    expected = params.copy(deep=True)

    pd.testing.assert_frame_equal(res, expected)
