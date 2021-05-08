from functools import partial

import pandas as pd

from src.policies.find_people_to_vaccinate import find_people_to_vaccinate


def baseline(sim_inputs):  # noqa: U100
    return {}


def no_vaccinations_after_feb_15(sim_inputs):
    start_date = sim_inputs["duration"]["start"]
    init_start = start_date - pd.Timedelta(31, unit="D")

    vaccination_func = sim_inputs["vaccination_models"]["standard"]["model"]
    vaccination_shares = vaccination_func.keywords["vaccination_shares"]
    vaccination_shares["2021-02-15":] = 0.0

    vaccination_func = partial(
        find_people_to_vaccinate,
        vaccination_shares=vaccination_shares,
        init_start=init_start,
    )
    vaccination_models = {"standard": {"model": vaccination_func}}
    return {"vaccination_models": vaccination_models}


def no_rapid_tests(sim_inputs):  # noqa: U100
    out = {
        "rapid_test_models": None,
        "rapid_test_reaction_models": None,
    }
    return out
