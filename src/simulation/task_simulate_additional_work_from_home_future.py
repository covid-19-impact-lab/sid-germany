"""Simulate different work from home scenarios into the future."""
import pandas as pd
import pytask
from sid import get_simulate_func

from src.config import BLD
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (  # noqa
    create_initial_conditions,
)
from src.policies.full_policy_blocks import get_hard_lockdown
from src.policies.full_policy_blocks import get_soft_lockdown_with_ab_schooling
from src.policies.policy_tools import combine_dictionaries

WFH_PARAMETRIZATION = []
FUTURE_WFH_SCENARIO_NAMES = ["baseline"]
WORK_MULTIPLIERS = [0.4]  ### Arbitrary

FUTURE_WFH_SEEDS = [15_000 + i for i in range(1)]  ###
for name, work_multiplier in zip(FUTURE_WFH_SCENARIO_NAMES, WORK_MULTIPLIERS):
    for seed in FUTURE_WFH_SEEDS:
        path = (
            BLD
            / "simulations"
            / "work_from_home_future"
            / f"{name}_{seed}"
            / "time_series"
        )
        spec = (work_multiplier, seed, path)
        WFH_PARAMETRIZATION.append(spec)


@pytask.mark.depends_on(
    {
        "initial_states": BLD / "data" / "debug_initial_states.parquet",  ###
        "share_known_cases": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases.pkl",
        "params": SRC / "simulation" / "estimated_params.pkl",
        "policy_py": SRC / "policies" / "combine_policies_over_periods.py",
        "contacts_py": SRC / "contact_models" / "get_contact_models.py",
    }
)
@pytask.mark.parametrize("work_multiplier, seed, produces", WFH_PARAMETRIZATION)
def task_simulate_work_from_home_scenario(depends_on, work_multiplier, seed, produces):
    start_date = pd.Timestamp("2021-01-05")  # - (7 days for RKI + reporting delay)
    end_date = pd.Timestamp("2021-01-15")  ###
    init_start = start_date - pd.Timedelta(31, unit="D")
    init_end = start_date - pd.Timedelta(1, unit="D")

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
    estimation_policies = _get_future_policies(
        contact_models=contact_models,
        work_multiplier=work_multiplier,
        start_date=start_date,
        end_date=end_date,
    )

    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=contact_models,
        contact_policies=estimation_policies,
        duration={"start": start_date, "end": end_date},
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


def _get_future_policies(contact_models, work_multiplier, start_date):
    """Get future policy scenario.

    Args:
        contact_models (dict)
        work_multiplier (float)
        start_date (pandas.Timestamp)

    Returns:
        policies (dict):

    """
    multipliers = {
        "educ": 0.6,
        "work": work_multiplier,
        "other": 0.4,
    }

    to_combine = [
        get_hard_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": str(start_date.date()),
                "end_date": "2021-02-06",
                "prefix": "hard_lockdown",
            },
            # fatigued other multiplier
            other_contacts_multiplier=0.50,
        ),
        get_soft_lockdown_with_ab_schooling(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-02-07",
                "end_date": "2021-04-01",
                "prefix": "maintain_low_infections",
            },
            multipliers=multipliers,
            age_cutoff=12,
        ),
    ]
    return combine_dictionaries(to_combine)
