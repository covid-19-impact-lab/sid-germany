from itertools import product

import pandas as pd
import pytask

from src.config import BLD
from src.simulation.scenario_config import (
    create_path_to_rapid_test_statistic_time_series,
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

RATES = [
    "false_negative_rate",
    "false_positive_rate",
    "true_negative_rate",
    "true_positive_rate",
]

RAPID_TEST_STATISTICS = []
for out, channel, share_type in product(OUTCOMES, CHANNELS, SHARE_TYPES):
    RAPID_TEST_STATISTICS.append(f"{share_type}_{out}_by_{channel}")
for out, channel in product(RATES, CHANNELS):
    RAPID_TEST_STATISTICS.append(f"{out}_by_{channel}")


_PARAMETRIZATION = [
    (
        column,
        create_path_to_rapid_test_statistic_time_series("spring_baseline", column),
    )
    for column in RAPID_TEST_STATISTICS
]


@pytask.mark.skipif(_N_SEEDS == 0, reason="spring baseline did not run.")
@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.parametrize("column, produces", _PARAMETRIZATION)
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


@pytask.mark.depends_on(_DEPENDENCIES)
def task_check_that_a_table_was_created_for_each_rapid_test_statistic(depends_on):
    statistics_saved_by_sid = pd.read_csv(depends_on[0]).columns
    to_skip = ["date", "n_individuals", "Unnamed: 0"]
    should_have_a_table = [x for x in statistics_saved_by_sid if x not in to_skip]
    assert set(should_have_a_table) == set(
        RAPID_TEST_STATISTICS
    ), "Some rapid test statistic columns that should have a table do not."


@pytask.mark.depends_on([path for col, path in _PARAMETRIZATION])
@pytask.mark.produces(BLD / "tables" / "rapid_test_statistics.csv")
def task_create_nice_rapid_test_statistic_table_for_lookup(produces):
    to_concat = [pd.read_pickle(path)[[column]] for column, path in _PARAMETRIZATION]
    df = pd.concat(to_concat, axis=1)
    df.round(4).to_csv(produces)
