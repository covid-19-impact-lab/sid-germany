import itertools
from src.config import BLD


SCENARIOS = ["optimistic", "pessimistic"]

CHRISTMAS_MODES = ["full", "same_group"]

CONTACT_TRACING_MULTIPLIERS = [None, 0.5, 0.1]

CARTESIAN_PRODUCT = list(itertools.product(
    SCENARIOS, CHRISTMAS_MODES, CONTACT_TRACING_MULTIPLIERS
))


def create_christmas_parametrization():
    paths = [create_output_path_for_simulation(*args) for args in CARTESIAN_PRODUCT]
    produces = [create_path_to_last_states(*args) for args in CARTESIAN_PRODUCT]

    return zip(*zip(*CARTESIAN_PRODUCT), paths, produces)


def create_output_path_for_simulation(
    scenario, christmas_mode, contact_tracing_multiplier
):
    ctm_str = (
        "wo_ct"
        if contact_tracing_multiplier is None
        else f"w_ct_{str(contact_tracing_multiplier).replace('.', '_')}"
    )
    path = (
        BLD / "simulation" / f"simulation_christmas_mode_{christmas_mode}_"
        f"{ctm_str}_{scenario}"
    )

    return path


def create_path_to_last_states(scenario, christmas_mode, contact_tracing_multiplier):
    path = create_output_path_for_simulation(
        scenario, christmas_mode, contact_tracing_multiplier
    )
    return path / "last_states" / "last_states.parquet"


def create_path_to_time_series(scenario, christmas_mode, contact_tracing_multiplier):
    path = create_output_path_for_simulation(
        scenario, christmas_mode, contact_tracing_multiplier
    )
    return path / "time_series"
