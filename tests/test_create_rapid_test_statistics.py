import numpy as np
import pandas as pd

from src.testing.create_rapid_test_statistics import (
    _calculate_true_positive_and_false_negatives,
)
from src.testing.create_rapid_test_statistics import _calculate_weights
from src.testing.create_rapid_test_statistics import _share_demanded_by_infected
from src.testing.create_rapid_test_statistics import create_rapid_test_statistics


def test_share_demanded_by_infected():
    # 1st: not infected. does not demand test -> should be ignored
    # 2nd: not infected and demands a test -> enters denominator
    # 3rd: infected and does not demand a test -> should be ignored
    # 4th and 5th: infected and demand a test -> enter numerator and denominator
    states = pd.DataFrame({"currently_infected": [False, False, True, True, True]})
    weights = pd.DataFrame({"channel": [0, 0.5, 0.2, 1, 0.5]})
    demand_by_channel = pd.DataFrame({"channel": [False, True, False, True, True]})
    res = _share_demanded_by_infected(
        demand_by_channel=demand_by_channel,
        states=states,
        weights=weights,
        channel="channel",
    )
    expected = 1.5 / (0.5 + 1.5)
    assert res == expected


def test_calculate_weights():
    demand_by_channel = pd.DataFrame(
        {
            "a": [False, False, True, True],
            "b": [False, True, False, True],
        }
    )
    expected = pd.DataFrame(
        {
            "a": [0, 0, 1, 0.5],
            "b": [0, 1, 0, 0.5],
        }
    )
    res = _calculate_weights(demand_by_channel)
    assert expected.equals(res)


def test_create_rapid_test_statistics(monkeypatch):
    date = pd.Timestamp("2021-04-26")
    demand_by_channel = pd.DataFrame(
        {
            "a": [False, False, True, True, False, False, True, True],
            "b": [False, True, False, True, False, True, False, True],
        }
    )
    states = pd.DataFrame(
        {
            "currently_infected": [False, False, True, True, False, False, False, True],
        }
    )

    def mocked_sample_test_outcome(states, receives_rapid_test, params, seed):
        out = pd.Series([True, False] * int(len(states) / 2), index=states.index)
        out[~receives_rapid_test] = False
        return out

    monkeypatch.setattr(
        "src.testing.create_rapid_test_statistics._sample_test_outcome",
        mocked_sample_test_outcome,
    )

    res = create_rapid_test_statistics(
        demand_by_channel=demand_by_channel,
        states=states,
        date=date,
        params=None,
    )

    # weights:
    # a: 0, 0, 1, 0.5, 0, 0, 1, 0.5
    # b: 0, 1, 0, 0.5, 0, 1, 0, 0.5
    #
    # groups:
    # a: 2, 3, 6, 7
    # b: 1, 3, 5, 7
    #
    # infected: 2, 3, 7
    #
    # test results overall
    # not tested: 0, 4
    # tested negative: 1, 3, 5, 7
    # tested positive: 2, 6
    #
    # true positive: 2
    # true negative: 1, 5
    # false negative: 3, 7
    # false positive: 6

    expected = pd.DataFrame(
        {
            0: {
                "date": date,
                "n_individuals": 8,
                "share_with_rapid_test_through_a": 3 / 8,
                "share_of_a_rapid_tests_demanded_by_infected": 2 / 3,
                "share_with_rapid_test_through_b": 3 / 8,
                "share_of_b_rapid_tests_demanded_by_infected": 1 / 3,
                "share_with_rapid_test": 0.75,
                "n_rapid_tests_overall": 6,
                "n_rapid_tests_through_a": 3,
                "n_rapid_tests_through_b": 3,
                # overall shares
                "true_positive_rate_overall": 0.5,
                "true_negative_rate_overall": 0.5,
                "false_negative_rate_overall": 0.5,
                "false_positive_rate_overall": 0.5,
                # shares in a
                "true_positive_rate_in_a": 0.5,
                "true_negative_rate_in_a": 0.0,
                "false_negative_rate_in_a": 1.0,
                "false_positive_rate_in_a": 0.5,
                # shares in b
                "true_positive_rate_in_b": np.nan,
                "true_negative_rate_in_b": 0.5,
                "false_negative_rate_in_b": 0.5,
                "false_positive_rate_in_b": np.nan,
            }
        }
    )
    assert set(expected.index) == set(res.index)
    pd.testing.assert_frame_equal(expected.loc[res.index], res)


def test_calculate_true_positive_and_false_negatives():
    # 1: True positive
    # 2: True negative
    # 3: False positive
    # 4 and 5: False negative
    # 6: not tested
    states = pd.DataFrame(
        {
            "currently_infected": [True, False, False, True, True, True],
        }
    )
    rapid_test_results = pd.Series([True, False, True, False, False, False])
    receives_rapid_test = pd.Series([True, True, True, True, True, False])

    (
        res_share_true_positive,
        res_share_false_negative,
    ) = _calculate_true_positive_and_false_negatives(
        states=states,
        rapid_test_results=rapid_test_results,
        receives_rapid_test=receives_rapid_test,
    )

    assert res_share_true_positive == 1 / 2  # 1 and 3 tested positive, 1 is infected
    assert res_share_false_negative == 2 / 3  # 2,4,5 tested negative, 4, 5 infected
