"""Simulate different work from home scenarios into the future.

=> wfh_share    stay_home_share     work_multiplier
   15%            (+5%) 20%             0.70
   25%            (+5%) 30%             0.55
   35%            (+5%) 40%             0.40
   55%            (+5%) 60%             0.10



"""
import pandas as pd
import pytask
from sid import get_simulate_func

from src.config import BLD
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (  # noqa
    create_initial_conditions,
)
from src.simulation.get_future_policies import get_future_policies


# ----------------------- To be configured ------------------------------

FUTURE_WFH_SEEDS = [800_000 * i for i in range(6)]
FUTURE_WFH_SCENARIO_NAMES = [
    "november_baseline",
    "mobility_data_baseline",
    "1st_lockdown_strict",
    "full_potential",
]

WORK_MULTIPLIERS = [
    0.70,
    0.55,
    0.40,
    0.10,
]


OTHER_MULTIPLIERS = [0.5] * len(WORK_MULTIPLIERS)
START_DATE = pd.Timestamp("2021-01-05")
END_DATE = pd.Timestamp("2021-03-15")

# ------------------------------------------------------------------------

WFH_PARAMETRIZATION = []
for name, work_multiplier, other_multiplier in zip(
    FUTURE_WFH_SCENARIO_NAMES, WORK_MULTIPLIERS, OTHER_MULTIPLIERS
):
    for seed in FUTURE_WFH_SEEDS:
        path = (
            BLD
            / "simulations"
            / "work_from_home_future_schools_open"
            / f"{name}_{seed}"
            / "time_series"
        )
        spec = (work_multiplier, other_multiplier, seed, path)
        WFH_PARAMETRIZATION.append(spec)


@pytask.mark.depends_on(
    {
        "initial_states": BLD / "data" / "initial_states.parquet",
        "share_known_cases": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases.pkl",
        "params": SRC / "simulation" / "estimated_params.pkl",
        "contacts_py": SRC / "contact_models" / "get_contact_models.py",
    }
)
@pytask.mark.parametrize(
    "work_multiplier, other_multiplier, seed, produces", WFH_PARAMETRIZATION
)
def task_simulate_work_from_home_scenario(
    depends_on, work_multiplier, other_multiplier, seed, produces
):
    init_start = START_DATE - pd.Timedelta(31, unit="D")
    init_end = START_DATE - pd.Timedelta(1, unit="D")

    initial_states = pd.read_parquet(depends_on["initial_states"])
    share_known_cases = pd.read_pickle(depends_on["share_known_cases"])
    params = pd.read_pickle(depends_on["params"])

    initial_conditions = create_initial_conditions(
        start=init_start,
        end=init_end,
        seed=3484,
        reporting_delay=5,
    )

    contact_models = get_all_contact_models(
        christmas_mode=None, n_extra_contacts_before_christmas=None
    )
    estimation_policies = get_future_policies(
        contact_models=contact_models,
        work_multiplier=work_multiplier,
        other_multiplier=other_multiplier,
        schools_open=True,
    )

    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=contact_models,
        contact_policies=estimation_policies,
        duration={"start": START_DATE, "end": END_DATE},
        initial_conditions=initial_conditions,
        share_known_cases=share_known_cases,
        path=produces.parent,
        seed=seed,
        saved_columns={
            "initial_states": ["age_group_rki"],
            "disease_states": ["newly_infected"],
            "time": ["date"],
            "other": ["new_known_case"],
        },
    )
    simulate(params)
