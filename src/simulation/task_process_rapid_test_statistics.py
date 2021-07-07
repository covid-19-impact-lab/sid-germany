import pandas as pd
import pytask

from src.config import BLD
from src.simulation.scenario_config import (
    create_path_to_rapid_test_statistic_time_series,
)
from src.simulation.scenario_config import create_path_to_raw_rapid_test_statistics
from src.simulation.scenario_config import get_named_scenarios

_N_SEEDS = get_named_scenarios()["combined_baseline"]["n_seeds"]

_DEPENDENCIES = {
    seed: create_path_to_raw_rapid_test_statistics("combined_baseline", seed)
    for seed in range(_N_SEEDS)
}

# private could additionally be split in "hh", "sym_without_pcr", "other_contact"
CHANNELS = ["private", "work", "educ"]
TYPES = ["true_positive", "false_positive", "true_negative", "false_negative"]

DEMAND_SHARE_COLS = [f"share_with_rapid_test_through_{c}" for c in CHANNELS] + [
    "share_with_rapid_test"
]
SHARE_INFECTED_COLS = [
    f"share_of_{c}_rapid_tests_demanded_by_infected" for c in CHANNELS
]
_SHARE_CORRECT_AND_FALSE_COLS = []
for typ in TYPES:
    _SHARE_CORRECT_AND_FALSE_COLS.append(f"{typ}_rate_overall")
    for channel in CHANNELS:
        col = f"{typ}_rate_in_{channel}"
        _SHARE_CORRECT_AND_FALSE_COLS.append(col)
OTHER_COLS = [f"n_rapid_tests_through_{c}" for c in CHANNELS] + [
    "n_rapid_tests_overall_in_germany",
    "false_positive_rate_in_the_population",
]


RAPID_TEST_STATISTICS = (
    DEMAND_SHARE_COLS + SHARE_INFECTED_COLS + _SHARE_CORRECT_AND_FALSE_COLS + OTHER_COLS
)
_PARAMETRIZATION = [
    (
        column,
        create_path_to_rapid_test_statistic_time_series("combined_baseline", column),
    )
    for column in RAPID_TEST_STATISTICS
]


@pytask.mark.skipif(_N_SEEDS == 0, reason="combined_baseline did not run.")
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
