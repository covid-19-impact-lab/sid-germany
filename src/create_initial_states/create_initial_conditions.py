import itertools as it

from src.config import POPULATION_GERMANY
from src.create_initial_states.create_initial_immunity import create_initial_immunity
from src.create_initial_states.create_initial_infections import (
    create_initial_infections,
)


def create_initial_conditions(
    start,
    end,
    seed,
    virus_shares,
    reporting_delay,
    synthetic_data,
    empirical_infections,
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
        synthetic_data (pandas.DataFrame): The synthetic population data set. Needs to
            contain 'county' and 'age_group_rki' as columns.
        empirical_infections (pandas.DataFrame): The index must contain 'date', 'county'
            and 'age_group_rki'. Must contain a column 'upscaled_newly_infected'.

    Returns:
        initial_conditions (dict): dictionary containing the initial infections and
            initial immunity.

    """
    seed = it.count(seed)
    empirical_infections = empirical_infections["upscaled_newly_infected"]

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
