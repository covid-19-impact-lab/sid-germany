"""Build the specifiation for the base prognosis."""
from src.config import BLD
from src.config import FAST_FLAG


def build_base_prognosis_parametrization():
    """Build the nested parametrization.

    Returns:
        nested_parametrization (dict): Keys are the names of the scenarios.
            Values are lists of tuples. For each seed there is one tuple.
            Each tuple consists of:
                1. the path where sid will save the time series data.
                2. the scenario specification consisting of the educ and
                   other multiplier and work_fill_value.
                3. the seed to be used by sid.

    """
    n_seeds = 1 if FAST_FLAG else 15

    base_scenario = {
        "educ_multiplier": 0.8,  # schools open Feb 15th
        "work_fill_value": 0.7,  # level of 1st half of January
        "other_multiplier": 0.45,  # arbitrary
    }
    nov_home_office = {**base_scenario, "work_fill_value": 0.8}
    spring_home_office = {**base_scenario, "work_fill_value": 0.6}
    schools_stay_closed = {**base_scenario, "educ_multiplier": 0.0}

    if FAST_FLAG:
        scenarios = {
            "base_scenario": base_scenario,
        }
    if not FAST_FLAG:
        scenarios = {
            "base_scenario": base_scenario,
            "november_home_office_level": nov_home_office,
            "spring_home_office_level": spring_home_office,
            "keep_schools_closed": schools_stay_closed,
        }

    nested_parametrization = {}
    for name, scenario in scenarios.items():
        nested_parametrization[name] = []
        for i in range(n_seeds):
            seed = 300_000 + 700_000 * i
            produces = BLD / "simulations" / f"{name}_{i}" / "time_series"
            nested_parametrization[name].append((produces, scenario, seed))

    return nested_parametrization
