import itertools as it

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.create_initial_states.create_initial_immunity import create_initial_immunity
from src.create_initial_states.create_initial_infections import (
    create_initial_infections,
)
from src.shared import load_dataset


def create_initial_conditions(
    start,
    end,
    seed,
    virus_shares,
    reporting_delay,
    synthetic_data_path,
    reported_infections_path=BLD / "data" / "processed_time_series" / "rki.pkl",
    population_size=POPULATION_GERMANY,
):
    """Create the initial conditions, initial_infections and initial_immunity.

    Args:
        start (str or pd.Timestamp): Start date for collection of initial
            infections.
        end (str or pd.Timestamp): End date for collection of initial
            infections and initial immunity.
        seed (int)
        virus_shares (dict): Keys are the names of the virus strains. Values are
            pandas.Series with a DatetimeIndex and the share among newly infected
            individuals on each day as value.
        reporting_delay (int): Number of days by which the reporting of cases is
            delayed. If given, later days are used to get the infections of the
            demanded time frame.
        synthetic_data_path (pathlib.Path or str): Path from which to load the
            snythetic data.
        reported_infections_path (pathlib.Path or str): Path from which to load the
            reported infections. The function expects a DataFrame with a column
            named "newly_infected".

    Returns:
        initial_conditions (dict): dictionary containing the initial infections and
            initial immunity.

    """
    seed = it.count(seed)
    empirical_infections = load_dataset(reported_infections_path)[
        "upscaled_newly_infected"
    ]
    synthetic_data = load_dataset(synthetic_data_path)[["county", "age_group_rki"]]

    initial_infections = create_initial_infections(
        empirical_infections=empirical_infections,
        synthetic_data=synthetic_data,
        start=start,
        end=end,
        reporting_delay=reporting_delay,
        seed=next(seed),
        virus_shares=virus_shares,
        population_size=population_size,
    )

    initial_immunity = create_initial_immunity(
        empirical_infections=empirical_infections,
        synthetic_data=synthetic_data,
        date=end,
        initial_infections=initial_infections,
        reporting_delay=reporting_delay,
        seed=next(seed),
        population_size=population_size,
    )
    return {
        "initial_infections": initial_infections,
        "initial_immunity": initial_immunity,
        # virus shares are already inside the initial infections so not included here.
    }
