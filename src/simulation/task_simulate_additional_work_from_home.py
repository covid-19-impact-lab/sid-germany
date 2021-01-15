"""Simulate different work from home (WFH) scenarios for Oct to mid Dec.

Summary of data on work from home:
    1. approx. 56% of workers could work from home.
    2. 27-45% (midpoint: 36%) of workers worked from home during the 1st lockdown.
    3. in June only 16-28% worked from home.
    4. in November and December only 14-17% worked from home.

.. warning::
    Remember, we assume essential workers always go to work.
    Our work multiplier is the share of non-essential workers who still have work
    contacts.

Our baseline (see ``_get_work_from_home_policies``):
    - >95% effective work contacts October 1-22
    - 70% effective work contacts October 23-31
    - 63% effective work contacts in November and December.

Assuming that there are no changes in hygiene standards and essential workers
continue to work normally, we can look at what happens when additional workers
work from home by increasing the threshold by 1.5x the change we want for the
whole working population. The 1.5x scaling is done by
``_get_work_from_home_policies``.

Our scenarios:
    1. baseline: no change
    2. 1_pct_more: 1 % point more of workers stay home
    3. return_to_1st_lockdown: Given that ~36% stayed home during the 1st
       lockdown and only 16% in Nov/Dec that means to return to the 1st lockdown
       20% points more of workers would have to stay home.
    4. Fully exploiting the potential for work from home (56%) the difference to
       what happened in Nov / Dec would be 56-16 = 40% points more workers in
       home office. This is not implemented at the moment because it would mean
       that some multipliers are below 0. This is because we are not separating
       hygiene measures from staying away from work at the moment and because
       essential workers are fixed to go to work even though they may well be
       able to work from home.

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
from src.policies.full_policy_blocks import get_german_reopening_phase
from src.policies.full_policy_blocks import get_soft_lockdown
from src.policies.policy_tools import combine_dictionaries

WFH_PARAMETRIZATION = []
WFH_SCENARIO_NAMES = [
    "baseline",
    "1_pct_more",
    "return_to_1st_lockdown",
]
WFH_SCENARIO_VALUES = [
    0,
    0.01,
    0.2,
]
WFH_SEEDS = [10_000 * i for i in range(2)]
for name, additional_work_from_home in zip(WFH_SCENARIO_NAMES, WFH_SCENARIO_VALUES):
    for seed in WFH_SEEDS:
        path = BLD / "simulations" / "work_from_home" / f"{name}_{seed}" / "time_series"
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
        "policy_py": SRC / "policies" / "combine_policies_over_periods.py",
        "contacts_py": SRC / "contact_models" / "get_contact_models.py",
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

    estimation_policies = _get_work_from_home_policies(
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


def _get_work_from_home_policies(contact_models, additional_work_from_home):
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


def _calculate_work_multiplier_accounting_for_essential_workers(
    participation_multiplier, hygiene_multiplier
):
    """Calculate the work_multiplier from the participation and hygiene multiplier.

    This assumes the share of essential workers is 0.33.

    Derivation:
        The work_multiplier is implemented as the share of non-systemically
        relevant workers that go to work and have (full risk) work contacts.
        Who goes to work is independent of how many work contacts someone has.

        => the work_multiplier can be interpreted as the share of (full risk)
           contacts that take place among the non-systemically relevant workers.

        => share_risk_contacts_still_happening = 0.33 + 0.66 * work_multiplier

        Another way to look at it is:

            share_risk_contacts_still_happening =
                participation_multiplier * hygiene_multiplier

        Combining the two ways of writing this, we get:

        0.33 + 0.66 * work_multiplier = participation_multiplier * hygiene_multiplier

        <=> 0.66 * work_multiplier =
                (participation_multiplier * hygiene_multiplier) - 0.33
        <=> work_multiplier =
                1.5 * (participation_multiplier * hygiene_multiplier) - 0.5


    Args:
        participation_multiplier (float)
        hygiene_multiplier (float)

    Returns:
        work_multiplier (float)

    """
    work_multiplier = 1.5 * (participation_multiplier * hygiene_multiplier) - 0.5
    return work_multiplier
