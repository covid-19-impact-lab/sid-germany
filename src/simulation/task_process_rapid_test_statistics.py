from itertools import product

import pandas as pd
import pytask

from src.config import BLD
from src.simulation.scenario_config import (
    create_path_to_rapid_test_statistic_time_series as get_ts_path,
)
from src.simulation.scenario_config import create_path_to_raw_rapid_test_statistics
from src.simulation.scenario_config import get_named_scenarios

_N_SEEDS = get_named_scenarios()["spring_baseline"]["n_seeds"]

_DEPENDENCIES = {
    seed: create_path_to_raw_rapid_test_statistics("spring_baseline", seed)
    for seed in range(_N_SEEDS)
}

CHANNELS = ["private", "work", "educ", "overall"]
OUTCOMES = [
    "false_negative",
    "false_positive",
    "tested_negative",
    "tested_positive",
    "true_negative",
    "true_positive",
    "tested",
]
SHARE_TYPES = ["number", "popshare", "testshare"]

RAPID_TEST_STATISTICS = []
for out, channel, share_type in product(OUTCOMES, CHANNELS, SHARE_TYPES):
    RAPID_TEST_STATISTICS.append(f"{share_type}_{out}_by_{channel}")

_SINGLE_COL_PARAMETRIZATION = [
    (column, get_ts_path("spring_baseline", column)) for column in RAPID_TEST_STATISTICS
]


@pytask.mark.skipif(_N_SEEDS == 0, reason="spring baseline did not run.")
@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.parametrize("column, produces", _SINGLE_COL_PARAMETRIZATION)
def task_process_rapid_test_statistics(depends_on, column, produces):
    dfs = {
        seed: pd.read_csv(path, parse_dates=["date"], index_col="date")
        for seed, path in depends_on.items()
    }
    for df in dfs.values():
        assert not df.index.duplicated().any(), (
            "Duplicates in a rapid test statistic DataFrame's index. "
            "The csv file must be deleted before every run."
        )
    df = pd.concat({seed: df[column] for seed, df in dfs.items()}, axis=1)
    df[column] = df.mean(axis=1).rolling(window=7, min_periods=1, center=False).mean()
    df.to_pickle(produces)


def _get_rate_parametrization(channels):
    rate_parametrization = []
    for channel in channels:
        rate_parametrization += [
            (
                f"true_positive_rate_by_{channel}",
                {
                    "numerator": get_ts_path(
                        "spring_baseline", f"number_true_positive_by_{channel}"
                    ),
                    "denominator": get_ts_path(
                        "spring_baseline", f"number_tested_positive_by_{channel}"
                    ),
                },
                get_ts_path("spring_baseline", f"true_positive_rate_by_{channel}"),
            ),
            (
                f"false_positive_rate_by_{channel}",
                {
                    "numerator": get_ts_path(
                        "spring_baseline", f"number_false_positive_by_{channel}"
                    ),
                    "denominator": get_ts_path(
                        "spring_baseline", f"number_tested_positive_by_{channel}"
                    ),
                },
                get_ts_path("spring_baseline", f"false_positive_rate_by_{channel}"),
            ),
            (
                f"true_negative_rate_by_{channel}",
                {
                    "numerator": get_ts_path(
                        "spring_baseline", f"number_true_negative_by_{channel}"
                    ),
                    "denominator": get_ts_path(
                        "spring_baseline", f"number_tested_negative_by_{channel}"
                    ),
                },
                get_ts_path("spring_baseline", f"true_negative_rate_by_{channel}"),
            ),
            (
                f"false_negative_rate_by_{channel}",
                {
                    "numerator": get_ts_path(
                        "spring_baseline", f"number_false_negative_by_{channel}"
                    ),
                    "denominator": get_ts_path(
                        "spring_baseline", f"number_tested_negative_by_{channel}"
                    ),
                },
                get_ts_path("spring_baseline", f"false_negative_rate_by_{channel}"),
            ),
        ]
    return rate_parametrization


_RATE_PARAMETRIZATION = _get_rate_parametrization(CHANNELS)


@pytask.mark.parametrize("name, depends_on, produces", _RATE_PARAMETRIZATION)
def task_create_rapid_test_statistic_ratios(name, depends_on, produces):
    numerator = pd.read_pickle(depends_on["numerator"])
    denominator = pd.read_pickle(depends_on["denominator"])

    seeds = list(range(_N_SEEDS))
    rate_df = pd.DataFrame()
    # needed for plotting single runs
    for s in seeds:
        smooth_num = numerator[s].rolling(window=7, min_periods=1, center=False).mean()
        smooth_denom = (
            denominator[s].rolling(window=7, min_periods=1, center=False).mean()
        )
        rate_df[s] = smooth_num / smooth_denom

    # it's important to first average and smooth and **then** divide to get rid of noise
    # before the division.
    rate_df[name] = (
        # use that the mean is created **after** the seeds have been added
        numerator[numerator.columns[-1]]
        / denominator[denominator.columns[-1]]
    )
    rate_df.to_pickle(produces)


_ALL_RAPID_TEST_STATISTICS = [path for col, path in _SINGLE_COL_PARAMETRIZATION] + [
    spec[-1] for spec in _RATE_PARAMETRIZATION
]


@pytask.mark.depends_on(_ALL_RAPID_TEST_STATISTICS)
@pytask.mark.produces(BLD / "tables" / "rapid_test_statistics.csv")
def task_create_nice_rapid_test_statistic_table_for_lookup(produces):
    column_names = [col for col, _ in _SINGLE_COL_PARAMETRIZATION] + [
        spec[0] for spec in _RATE_PARAMETRIZATION
    ]
    assert len(set(column_names)) == len(column_names), (
        "There are duplicate names in the rapid test statistic columns. "
        "You probably forgot to specify a channel as part of the column name."
    )

    to_concat = [
        pd.read_pickle(path)[[column]] for column, path in _SINGLE_COL_PARAMETRIZATION
    ] + [pd.read_pickle(path)[[column]] for column, _, path in _RATE_PARAMETRIZATION]
    df = pd.concat(to_concat, axis=1)
    df.round(4).to_csv(produces)


@pytask.mark.depends_on(_DEPENDENCIES)
def task_check_that_a_table_was_created_for_each_rapid_test_statistic(depends_on):
    statistics_saved_by_sid = pd.read_csv(depends_on[0]).columns
    to_skip = ["date", "n_individuals", "Unnamed: 0"]
    should_have_a_table = [x for x in statistics_saved_by_sid if x not in to_skip]
    assert set(should_have_a_table) == set(
        RAPID_TEST_STATISTICS
    ), "Some rapid test statistic columns that should have a table do not."
