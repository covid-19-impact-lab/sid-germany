import pandas as pd
import pytask
from sid import get_simulate_func

from src.config import BLD
from src.config import SRC
from src.contact_models.get_contact_models import get_all_contact_models
from src.create_initial_states.create_initial_conditions import (  # noqa
    create_initial_conditions,
)
from src.policies.full_policy_blocks import get_german_reopening_phase
from src.policies.full_policy_blocks import get_soft_lockdown
from src.policies.policy_tools import combine_dictionaries

# additional_work_from_home, seed, time_series path
WFH_PARAMETRIZATION = []
WFH_SCENARIO_NAMES = ["baseline", "1pct_more"]
WFH_SCENARIO_VALUES = [0, 0.01]
WFH_SEEDS = [10_000, 20_000]
for name, additional_work_from_home in zip(WFH_SCENARIO_NAMES, WFH_SCENARIO_VALUES):
    for seed in WFH_SEEDS:
        path = BLD / "work_from_home" / f"{name}_{seed}"
        spec = (additional_work_from_home, seed, path)
        WFH_PARAMETRIZATION.append(spec)


@pytask.mark.depends_on(
    {
        "initial_states": BLD / "data" / "initial_states.parquet",
        "share_known_cases": BLD
        / "data"
        / "processed_time_series"
        / "share_known_cases.pkl",
        "params": SRC / "simulation" / "estimated_params.pkl",
        "contact_models": SRC / "contact_models" / "get_contact_models.py",
    }
)
@pytask.mark.parametrize(
    "additional_work_from_home, seed, produces", WFH_PARAMETRIZATION
)
def task_simulate_work_from_home_scenario(
    depends_on, additional_work_from_home, seed, produces
):
    start_date = pd.Timestamp("2020-10-01")
    end_date = pd.Timestamp("2020-12-12")
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

    estimation_policies = _get_estimation_policies(
        contact_models, additional_work_from_home
    )

    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=contact_models,
        contact_policies=estimation_policies,
        duration={"start": start_date, "end": end_date},
        initial_conditions=initial_conditions,
        share_known_cases=share_known_cases,
        path=produces,
        seed=seed,
        saved_columns={
            "initial_states": ["age_group_rki"],
            "disease_states": ["newly_infected"],
            "time": ["date"],
            "other": ["new_known_case"],
        },
    )
    simulate(params)


def _get_estimation_policies(contact_models, additional_work_from_home=0.0):
    """Get estimation policies from July to December 20th.

    Args:
        contact_models (dict): contact models
        additional_work_from_home (float): percentage points of additional
            workers that work from home. This change is rescaled into a
            change in the work priority threshold which only affects
            non-essential workers. Must lie between -0.66 and 0.66.

    """
    assert (
        -0.66 < additional_work_from_home < 0.66
    ), "additional_work_from_home outside (-0.66, 0.66)"

    work_reduction = 1.5 * additional_work_from_home
    reopening_start_multipliers = {
        "educ": 0.8,
        "work": 0.55 - work_reduction,
        "other": 0.45,
    }
    reopening_end_multipliers = {
        "educ": 0.8,
        "work": 0.95 - work_reduction,
        "other": 0.7,
    }
    anticipate_lockdown_multipliers = {
        "educ": 0.8,
        "work": 0.55 - work_reduction,
        "other": 0.5,
    }
    lockdown_light_multipliers = {
        "educ": 0.6,
        "work": 0.45 - work_reduction,
        "other": 0.4,
    }
    lockdown_light_multipliers_with_fatigue = {
        "educ": 0.6,
        "work": 0.45 - work_reduction,
        "other": 0.50,
    }

    to_combine = [
        get_german_reopening_phase(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-07-01",
                "end_date": "2020-10-22",
                "prefix": "reopening",
            },
            start_multipliers=reopening_start_multipliers,
            end_multipliers=reopening_end_multipliers,
            educ_switching_date="2020-08-01",
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-10-23",
                "end_date": "2020-11-01",
                "prefix": "anticipate_lockdown_light",
            },
            multipliers=anticipate_lockdown_multipliers,
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
                "end_date": "2020-12-20",
                "prefix": "lockdown_light_with_fatigue",
            },
            multipliers=lockdown_light_multipliers_with_fatigue,
        ),
    ]

    return combine_dictionaries(to_combine)
