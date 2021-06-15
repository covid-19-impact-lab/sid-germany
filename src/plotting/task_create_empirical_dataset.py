import pandas as pd
import pytask

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.config import BLD
from src.config import SRC


_DEPENDENCIES = {
    "rki": BLD / "data" / "processed_time_series" / "rki.pkl",
    "cosmo_ever_rapid_test": SRC
    / "original_data"
    / "testing"
    / "cosmo_share_ever_had_a_rapid_test.csv",
    "cosmo_weekly_rapid_test_last_4_weeks": SRC
    / "original_data"
    / "testing"
    / "cosmo_selftest_frequency_last_four_weeks.csv",
    "virus_shares_dict": BLD / "data" / "virus_strains" / "virus_shares_dict.pkl",
    "calculate_moments.py": SRC / "calculate_moments.py",
    "vaccinations": BLD / "data" / "vaccinations" / "vaccination_shares_raw.pkl",
    "r_effective": BLD / "data" / "processed_time_series" / "r_effective.pkl",
}


@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.produces(
    {
        "pkl": BLD / "data" / "empirical_data_for_plotting.pkl",
        "csv": BLD / "tables" / "empirical_analogues.csv",
    }
)
def task_create_empirical_dataset(depends_on, produces):
    rki = pd.read_pickle(depends_on["rki"])
    new_known_case = 7 * smoothed_outcome_per_hundred_thousand_rki(
        df=rki,
        outcome="newly_infected",
        take_logs=False,
        window=7,
    )
    new_known_case.name = "new_known_case"

    newly_deceased = 7 * smoothed_outcome_per_hundred_thousand_rki(
        df=rki,
        outcome="newly_deceased",
        take_logs=False,
        window=7,
    )
    newly_deceased.name = "newly_deceased"

    share_ever_rapid_test = pd.read_csv(
        depends_on["cosmo_ever_rapid_test"], parse_dates=["date"], index_col="date"
    )["share_ever_had_a_rapid_test"]
    share_ever_rapid_test.name = "share_ever_rapid_test"

    rapid_test_frequency = pd.read_csv(
        depends_on["cosmo_weekly_rapid_test_last_4_weeks"],
        parse_dates=["date"],
        index_col="date",
    )
    share_rapid_test_in_last_week = rapid_test_frequency[
        [
            "share_more_than_5_tests_per_week",
            "share_5_tests_per_week",
            "share_2-4_tests_per_week",
            "share_weekly",
        ]
    ].sum(axis=1)
    share_rapid_test_in_last_week.name = "share_rapid_test_in_last_week"

    share_b117 = pd.read_pickle(depends_on["virus_shares_dict"])["b117"]["2021-01-15":]
    share_b117.name = "share_b117"

    vacc_shares = pd.read_pickle(depends_on["vaccinations"]).sort_index().cumsum()
    vacc_shares.name = "ever_vaccinated"

    r_effective = pd.read_pickle(depends_on["r_effective"])
    r_effective.name = "r_effective"

    df = pd.concat(
        [
            new_known_case,
            newly_deceased,
            share_ever_rapid_test,
            share_rapid_test_in_last_week,
            share_b117,
            vacc_shares,
            r_effective,
        ],
        axis=1,
    )
    df.to_pickle(produces["pkl"])
    df.round(3).to_csv(produces["csv"])
