import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.create_initial_states.create_initial_immunity import create_initial_immunity
from src.create_initial_states.create_initial_infections import (
    create_initial_infections,
)

DEPENDENCIES = {
    "initial_infections.py": SRC
    / "create_initial_states"
    / "create_initial_infections.py",
    "initial_immunity.py": SRC / "create_initial_states" / "create_initial_immunity.py",
    "rki": BLD / "data" / "processed_time_series" / "rki.pkl",
    "initial_states": BLD / "data" / "inital_states.parquet",
}

PRODUCTS = {
    "initial_infections": BLD / "example_simulation" / "initial_infections.pkl",
    "initial_immunity": BLD / "example_simulation" / "initial_immunity.pkl",
}


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.produces(PRODUCTS)
def task_create_initial_conditions(depends_on, produces):
    estimation_start = "2020-08-02"
    empirical_data = pd.read_pickle(depends_on["rki"])["newly_infected"]
    initial_states = pd.read_parquet(depends_on["initial_states"])

    initial_infections = create_initial_infections(
        empirical_data=empirical_data,
        synthetic_data=initial_states,
        start=empirical_data.index.get_level_values("date").min(),
        end=estimation_start,
        undetected_multiplier=4,
        seed=4484,
    )
    initial_infections.to_pickle(produces["initial_infections"])

    initial_immunity = create_initial_immunity(
        empirical_data=empirical_data,
        synthetic_data=initial_states,
        date=estimation_start,
        initial_infections=initial_infections,
        undetected_multiplier=4,
        seed=94,
    )
    initial_immunity.to_pickle(produces["initial_immunity"])
