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
}


@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.produces(BLD / "data" / "empirical_data_for_plotting.pkl")
def task_create_empirical_dataset(depends_on, produces):
    rki = pd.read_pickle(depends_on["rki"])
    new_known_case = 7 * smoothed_outcome_per_hundred_thousand_rki(
        df=rki,
        outcome="newly_infected",
        take_logs=False,
        window=7,
    )

    newly_deceased = 7 * smoothed_outcome_per_hundred_thousand_rki(
        df=rki,
        outcome="newly_deceased",
        take_logs=False,
        window=7,
    )

    share_ever_rapid_test = pd.read_csv(
        depends_on["cosmo_ever_rapid_test"], parse_dates=["date"], index_col="date"
    )
    share_rapid_test_in_last_week = pd.read_csv(
        depends_on["cosmo_weekly_rapid_test_last_4_weeks"],
        parse_dates=["date"],
        index_col="date",
    )
    share_b117 = pd.read_pickle(depends_on["virus_shares_dict"])["b117"]

    df = pd.concat(
        [
            new_known_case,
            newly_deceased,
            share_ever_rapid_test,
            share_rapid_test_in_last_week,
            share_b117,
        ]
    )
    df.to_pickle(produces)
