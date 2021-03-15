import itertools as it

from src.config import BLD
from src.create_initial_states.create_initial_immunity import create_initial_immunity
from src.create_initial_states.create_initial_infections import (
    create_initial_infections,
)
from src.shared import load_dataset


def create_initial_conditions(
    start,
    end,
    seed,
    reporting_delay=0,
    synthetic_data_path=BLD / "data" / "initial_states.parquet",
    reported_infections_path=BLD / "data" / "processed_time_series" / "rki.pkl",
    virus_shares=None,
):
    """Create the initial conditions, initial_infections and initial_immunity.

    Args:
        start (str or pd.Timestamp): Start date for collection of initial
            infections.
        end (str or pd.Timestamp): End date for collection of initial
            infections and initial immunity.
        seed (int)
        reporting_delay (int): Number of days by which the reporting of cases is
            delayed. If given, later days are used to get the infections of the
            demanded time frame.
        synthetic_data_path (pathlib.Path or str): Path from which to load the
            snythetic data.
        reported_infections_path (pathlib.Path or str): Path from which to load the
            reported infections. The function expects a DataFrame with a column
            named "newly_infected".
        virus_shares (dict, optional): A mapping between the names
            of the virus strains and their share among newly infected
            individuals over time. If None, the b117 data is loaded as only strain.

    Returns:
        initial_conditions (dict): dictionary containing the initial infections and
            initial immunity.

    """
    seed = it.count(seed)
    empirical_data = load_dataset(reported_infections_path)["upscaled_newly_infected"]
    synthetic_data = load_dataset(synthetic_data_path)[["county", "age_group_rki"]]
    if virus_shares is None:
        b117 = load_dataset(BLD / "data" / "virus_strains" / "b117.pkl")
        virus_shares = {"base_strain": 1 - b117, "b117": b117}

    initial_infections = create_initial_infections(
        empirical_data=empirical_data,
        synthetic_data=synthetic_data,
        start=start,
        end=end,
        reporting_delay=reporting_delay,
        seed=next(seed),
        virus_shares=virus_shares,
    )

    initial_immunity = create_initial_immunity(
        empirical_data=empirical_data,
        synthetic_data=synthetic_data,
        date=end,
        initial_infections=initial_infections,
        reporting_delay=reporting_delay,
        seed=next(seed),
    )
    return {
        "initial_infections": initial_infections,
        "initial_immunity": initial_immunity,
        # virus shares are already inside the initial infections so not included here.
    }
