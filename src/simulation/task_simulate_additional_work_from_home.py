"""Simulate different work from home (WFH) scenarios for November to mid December.

Summary of data on work from home:
    1. in November and December only 14-17% worked from home.
    2. 25-35% of workers worked from home during the 1st lockdown.
    3. approx. 56% of workers could work from home.

work_multiplier = (1 - stay_home_share * 1.5) * 0.95

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
from src.policies.full_policy_blocks import get_soft_lockdown
from src.policies.policy_tools import combine_dictionaries

WFH_SEEDS = [1_000_000 * i for i in range(48)]

WFH_PARAMETRIZATION = []
WFH_SCENARIO_NAMES = [
    "baseline",
    "1_pct_more",
    "1st_lockdown_weak",
    "1st_lockdown_strict",
    "full_potential",
]
WFH_WORK_MULTIPLIERS = [
    (0.73, 0.76),
    (0.715, 0.745),  # 1 pct more
    (0.625, 0.625),  # 1st lockdown weak
    (0.475, 0.475),  # 1st lockdown strict
    (0.1, 0.1),  # full potential
]
for name, work_multipliers in zip(WFH_SCENARIO_NAMES, WFH_WORK_MULTIPLIERS):
    for seed in WFH_SEEDS:
        path = BLD / "simulations" / "work_from_home" / f"{name}_{seed}" / "time_series"
        spec = (work_multipliers, seed, path)
        WFH_PARAMETRIZATION.append(spec)


@pytask.mark.depends_on(
    {
        "initial_states": BLD / "data" / "initial_states.parquet",
        "share_known_cases": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases.pkl",
        "params": SRC / "simulation" / "estimated_params.pkl",
        "policy_py": SRC / "policies" / "combine_policies_over_periods.py",
        "contacts_py": SRC / "contact_models" / "get_contact_models.py",
    }
)
@pytask.mark.parametrize("work_multipliers, seed, produces", WFH_PARAMETRIZATION)
def task_simulate_work_from_home_scenario(depends_on, work_multipliers, seed, produces):
    start_date = pd.Timestamp("2020-10-01")
    end_date = pd.Timestamp("2020-12-15")
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

    estimation_policies = _get_work_from_home_policies(contact_models, work_multipliers)

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


def _get_work_from_home_policies(contact_models, work_multipliers):
    """Get estimation policies from November to December 15th."""
    pre_fall_vacation_multipliers = {"educ": 0.8, "work": 0.775, "other": 0.75}
    fall_vacation_multipliers = {"educ": 0.8, "work": 0.63, "other": 1.0}
    post_fall_vacation_multipliers = {"educ": 0.8, "work": 0.775, "other": 0.65}
    # anticipate_lockdown_multipliers = {"educ": 0.8, "work": 0.55, "other": 0.5}
    lockdown_light_multipliers = {"educ": 0.6, "work": work_multipliers[0] * 0.95, "other": 0.45}
    print(work_multipliers)
    print(lockdown_light_multipliers)
    lockdown_light_multipliers_with_fatigue = {"educ": 0.6, "work": work_multipliers[1] * 0.95, "other": 0.55}
    to_combine = [
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-01",
                "end_date": "2020-10-09",
                "prefix": "pre_fall_vacation",
            },
            multipliers=pre_fall_vacation_multipliers,
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-10",
                "end_date": "2020-10-23",
                "prefix": "fall_vacation",
            },
            multipliers=fall_vacation_multipliers,
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-24",
                "end_date": "2020-11-01",
                "prefix": "post_fall_vacation",
            },
            multipliers=post_fall_vacation_multipliers,
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-02",
                "end_date": "2020-11-22",
                "prefix": "lockdown_light",
            },
            multipliers=lockdown_light_multipliers,
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-11-23",
                "end_date": "2020-12-24",
                "prefix": "lockdown_light_with_fatigue",
            },
            multipliers=lockdown_light_multipliers_with_fatigue,
        ),
    ]


    return combine_dictionaries(to_combine)
