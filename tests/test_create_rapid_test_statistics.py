import pandas as pd
from pandas.testing import assert_series_equal

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
                # numbers
                "number_false_negative_by_a": 2 * scaling,
                "number_false_negative_by_b": 2 * scaling,
                "number_false_negative_by_overall": 2 * scaling,
                "number_true_negative_by_a": 0.0,
                "number_true_negative_by_b": 2 * scaling,
                "number_true_negative_by_overall": 2 * scaling,
                "number_false_positive_by_a": 1 * scaling,
                "number_false_positive_by_b": 0.0,
                "number_false_positive_by_overall": 1 * scaling,
                "number_true_positive_by_a": 1 * scaling,
                "number_true_positive_by_b": 0.0,
                "number_true_positive_by_overall": 1 * scaling,
                "number_tested_by_a": 4 * scaling,
                "number_tested_by_b": 4 * scaling,
                "number_tested_by_overall": 6 * scaling,
                "number_tested_negative_by_a": 2 * scaling,
                "number_tested_negative_by_b": 4 * scaling,
                "number_tested_negative_by_overall": 4 * scaling,
                "number_tested_positive_by_a": 2 * scaling,
                "number_tested_positive_by_b": 0 * scaling,
                "number_tested_positive_by_overall": 2 * scaling,
                # popshare
                "popshare_false_negative_by_a": 2 / len(states),
                "popshare_false_negative_by_b": 2 / len(states),
                "popshare_false_negative_by_overall": 2 / len(states),
                "popshare_true_negative_by_a": 0.0,
                "popshare_true_negative_by_b": 2 / len(states),
                "popshare_true_negative_by_overall": 2 / len(states),
                "popshare_false_positive_by_a": 1 / len(states),
                "popshare_false_positive_by_b": 0.0,
                "popshare_false_positive_by_overall": 1 / len(states),
                "popshare_true_positive_by_a": 1 / len(states),
                "popshare_true_positive_by_b": 0.0,
                "popshare_true_positive_by_overall": 1 / len(states),
                "popshare_tested_by_a": 4 / len(states),
                "popshare_tested_by_b": 4 / len(states),
                "popshare_tested_by_overall": 6 / len(states),
                "popshare_tested_negative_by_a": 2 / len(states),
                "popshare_tested_negative_by_b": 4 / len(states),
                "popshare_tested_negative_by_overall": 4 / len(states),
                "popshare_tested_positive_by_a": 2 / len(states),
                "popshare_tested_positive_by_b": 0 / len(states),
                "popshare_tested_positive_by_overall": 2 / len(states),
                # testshare
                "testshare_false_negative_by_a": 2 / 4,
                "testshare_false_negative_by_b": 2 / 4,
                "testshare_false_negative_by_overall": 2 / 6,
                "testshare_true_negative_by_a": 0.0,
                "testshare_true_negative_by_b": 2 / 4,
                "testshare_true_negative_by_overall": 2 / 6,
                "testshare_false_positive_by_a": 1 / 4,
                "testshare_false_positive_by_b": 0 / 4,
                "testshare_false_positive_by_overall": 1 / 6,
                "testshare_true_positive_by_a": 1 / 4,
                "testshare_true_positive_by_b": 0.0,
                "testshare_true_positive_by_overall": 1 / 6,
                "testshare_tested_by_a": 4 / 4,
                "testshare_tested_by_b": 4 / 4,
                "testshare_tested_by_overall": 6 / 6,
                "testshare_tested_negative_by_a": 2 / 4,
                "testshare_tested_negative_by_b": 4 / 4,
                "testshare_tested_negative_by_overall": 4 / 6,
                "testshare_tested_positive_by_a": 2 / 4,
                "testshare_tested_positive_by_b": 0 / 4,
                "testshare_tested_positive_by_overall": 2 / 6,
            }
        }
    )
    expected.index.name = "index"
    pd.testing.assert_frame_equal(expected.sort_index(), res.sort_index())


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

    scaling = POPULATION_GERMANY / len(states)

    res = pd.Series(
        _calculate_rapid_test_statistics_by_channel(
            states=states,
            rapid_test_results=rapid_test_results,
            receives_rapid_test=receives_rapid_test,
            channel_name="channel",
        )
    )

    expected = pd.Series(
        {
            "number_tested_by_channel": 5 * scaling,
            "number_tested_positive_by_channel": 2 * scaling,
            "number_tested_negative_by_channel": 3 * scaling,
            "number_false_positive_by_channel": 1 * scaling,
            "number_false_negative_by_channel": 2 * scaling,
            "number_true_positive_by_channel": 1 * scaling,
            "number_true_negative_by_channel": 1 * scaling,
            "popshare_tested_by_channel": 5 / len(states),
            "popshare_tested_positive_by_channel": 2 / len(states),
            "popshare_tested_negative_by_channel": 3 / len(states),
            "popshare_false_positive_by_channel": 1 / len(states),
            "popshare_false_negative_by_channel": 2 / len(states),
            "popshare_true_positive_by_channel": 1 / len(states),
            "popshare_true_negative_by_channel": 1 / len(states),
            "testshare_tested_by_channel": 5 / 5,
            "testshare_tested_positive_by_channel": 2 / 5,
            "testshare_tested_negative_by_channel": 3 / 5,
            "testshare_false_positive_by_channel": 1 / 5,
            "testshare_false_negative_by_channel": 2 / 5,
            "testshare_true_positive_by_channel": 1 / 5,
            "testshare_true_negative_by_channel": 1 / 5,
        }
    )
    assert_series_equal(res.loc[expected.index], expected, check_names=False)
