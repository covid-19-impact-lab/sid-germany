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
    undetected_multiplier,
    seed,
    reporting_delay=0,
    synthetic_data_path=BLD / "data" / "initial_states.parquet",
    reported_infections_path=BLD / "data" / "processed_time_series" / "rki.pkl",
):
    """Create the initial conditions, initial_infections and initial_immunity.

    Args:
        start (str or pd.Timestamp): Start date for collection of initial
            infections.
        end (str or pd.Timestamp): End date for collection of initial
            infections and initial immunity.
        undetected_multiplier (float): Multiplier used to scale up the observed
            infections to account for unknown cases. Must be >=1.
        seed (int)
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
    empirical_data = load_dataset(reported_infections_path)["newly_infected"]
    synthetic_data = load_dataset(synthetic_data_path)

    initial_infections = create_initial_infections(
        empirical_data=empirical_data,
        synthetic_data=synthetic_data,
        start=start,
        end=end,
        undetected_multiplier=undetected_multiplier,
        reporting_delay=reporting_delay,
        seed=next(seed),
    )

    initial_immunity = create_initial_immunity(
        empirical_data=empirical_data,
        synthetic_data=synthetic_data,
        date=end,
        initial_infections=initial_infections,
        undetected_multiplier=undetected_multiplier,
        reporting_delay=reporting_delay,
        seed=next(seed),
    )
    return {
        "initial_infections": initial_infections,
        "initial_immunity": initial_immunity,
    }