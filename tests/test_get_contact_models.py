import pandas as pd

from src.contact_models.get_contact_models import get_christmas_contact_models


def test_christmas_mode_full():
    res = get_christmas_contact_models("full", 2)
    expected = {
        "holiday_preparation": {"n_contacts": 2},
        "christmas_full_0": {
            "group_col": "christmas_group_id_0",
            "dates": [pd.Timestamp("2020-12-24")],
        },
        "christmas_full_1": {
            "group_col": "christmas_group_id_1",
            "dates": [pd.Timestamp("2020-12-25")],
        },
        "christmas_full_2": {
            "group_col": "christmas_group_id_2",
            "dates": [pd.Timestamp("2020-12-26")],
        },
    }
    assert res.keys() == expected.keys()
    for key, expected_kwargs in expected.items():
        res_kwargs = res[key]["model"].keywords
        assert res_kwargs == expected_kwargs


def test_christmas_mode_same_group():
    res = get_christmas_contact_models("same_group", 2)
    expected = {
        "holiday_preparation": {"n_contacts": 2},
        "christmas_same_group": {
            "group_col": "christmas_group_id_0",
            "dates": [
                pd.Timestamp("2020-12-24"),
                pd.Timestamp("2020-12-25"),
                pd.Timestamp("2020-12-26"),
            ],
        },
    }
    assert res.keys() == expected.keys()
    for key, expected_kwargs in expected.items():
        res_kwargs = res[key]["model"].keywords
        assert res_kwargs == expected_kwargs


def test_christmas_mode_meet_twice():
    res = get_christmas_contact_models("meet_twice", 3)
    expected = {
        "holiday_preparation": {"n_contacts": 3},
        "christmas_meet_twice": {
            "group_col": "christmas_group_id_0",
            "dates": [pd.Timestamp("2020-12-24"), pd.Timestamp("2020-12-25")],
        },
    }
    assert res.keys() == expected.keys()
    for key, expected_kwargs in expected.items():
        res_kwargs = res[key]["model"].keywords
        assert res_kwargs == expected_kwargs
