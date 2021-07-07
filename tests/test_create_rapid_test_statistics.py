import numpy as np
import pandas as pd

from src.config import POPULATION_GERMANY
from src.testing.create_rapid_test_statistics import (
    _calculate_rapid_test_statistics_by_channel,
)
from src.testing.create_rapid_test_statistics import create_rapid_test_statistics


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

    scaling = POPULATION_GERMANY / len(states)
    expected = pd.DataFrame(
        {
            0: {
                "date": date,
                # overall shares
                "true_positive_rate_by_overall": 0.5,
                "true_negative_rate_by_overall": 0.5,
                "false_negative_rate_by_overall": 0.5,
                "false_positive_rate_by_overall": 0.5,
                # shares in a
                "true_positive_rate_by_a": 0.5,
                "true_negative_rate_by_a": 0.0,
                "false_negative_rate_by_a": 1.0,
                "false_positive_rate_by_a": 0.5,
                # shares in b
                "true_positive_rate_by_b": np.nan,
                "true_negative_rate_by_b": 0.5,
                "false_negative_rate_by_b": 0.5,
                "false_positive_rate_by_b": np.nan,
                # numbers
                "number_false_negative_by_a": 2 * scaling,
                "number_false_negative_by_b": 2 * scaling,
                "number_false_negative_by_overall": 2 * scaling,
                "number_true_negative_by_a": 0.0,
                "number_true_negative_by_b": 2 * scaling,
                "number_true_negative_by_overall": 2 * scaling,
                "number_false_positive_by_a": 1 * scaling,
                "number_false_positive_by_b": np.nan,
                "number_false_positive_by_overall": 1 * scaling,
                "number_true_positive_by_a": 1 * scaling,
                "number_true_positive_by_b": 0.0,
                "number_true_positive_by_overall": 1 * scaling,
                #
                "number_tested_by_a": 4 * scaling,
                "number_tested_by_b": 4 * scaling,
                "number_tested_by_overall": 6 * scaling,
                "number_tested_negative_by_a": 2 * scaling,
                "number_tested_negative_by_b": 4 * scaling,
                "number_tested_negative_by_overall": 4 * scaling,
                "number_tested_positive_by_a": 2 * scaling,
                "number_tested_positive_by_b": 0 * scaling,
                "number_tested_positive_by_overall": 2 * scaling,
            }
        }
    )

    expected.columns = [1]
    df = pd.concat([res, expected], axis=1)
    diff = df[df[0] != df[1]].dropna()

    ### pd.testing.assert_frame_equal(expected, res.loc[expected.index])
    ### assert set(expected.index) == set(res.index)


def test_calculate_rapid_test_statistics_by_channel():
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

    res = _calculate_rapid_test_statistics_by_channel(
        states=states,
        rapid_test_results=rapid_test_results,
        receives_rapid_test=receives_rapid_test,
        channel_name="channel",
    )

    scaling = POPULATION_GERMANY / len(states)
    expected = {
        "number_tested_by_channel": 5 * scaling,
        "number_tested_positive_by_channel": 2 * scaling,
        "number_tested_negative_by_channel": 3 * scaling,
        "number_tested_false_positive_by_channel": 1 * scaling,
        "number_tested_false_negative_by_channel": 2 * scaling,
        "number_tested_true_positive_by_channel": 1 * scaling,
        "number_tested_true_negative_by_channel": 1 * scaling,
    }

    assert res == expected
