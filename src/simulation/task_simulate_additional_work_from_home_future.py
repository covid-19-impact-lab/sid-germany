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
from src.policies.full_policy_blocks import get_soft_lockdown
from src.policies.policy_tools import combine_dictionaries

# ----------------------- To be configured ------------------------------

FUTURE_WFH_SEEDS = [15_000 + i for i in range(1)]
FUTURE_WFH_SCENARIO_NAMES = ["baseline"]
WORK_MULTIPLIERS = [0.4]
OTHER_MULTIPLIERS = [0.4]
START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2021-01-15")

# ------------------------------------------------------------------------

WFH_PARAMETRIZATION = []
for name, work_multiplier, other_multiplier in zip(
    FUTURE_WFH_SCENARIO_NAMES, WORK_MULTIPLIERS, OTHER_MULTIPLIERS
):
    for seed in FUTURE_WFH_SEEDS:
        path = (
            BLD
            / "simulations"
            / "work_from_home_future"
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
        "work_days": BLD / "policies" / "google_workday_data.csv",
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
    google_data = pd.read_csv(depends_on["work_days"])
    google_data.index = pd.DatetimeIndex(google_data["date"])

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
        other_multiplier=other_multiplier,
        start_date=START_DATE,
        end_date=END_DATE,
        google_data=google_data,
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


def _get_future_policies(
    contact_models, work_multiplier, other_multiplier, google_data
):
    """Get future policy scenario.

    Args:
        contact_models (dict)
        work_multiplier (float): work multiplier used starting January 12th
        other_multiplier (float): other multiplier used starting January 12th
        google_data (pandas.DataFrame): google data to get the work multiplier
            of the 2nd January week.

    Returns:
        policies (dict):

    """
    work_change = google_data.loc["2021-01-04":"2021-01-11", "workplaces"].mean()
    second_jan_week_work_multiplier = 1 + work_change / 100
    multipliers = {
        "educ": 0.6,
        "work": work_multiplier,
        "other": other_multiplier,
    }

    to_combine = [
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2020-12-27",
                "end_date": "2021-01-03",
                "prefix": "post-christmas-lockdown",
            },
            multipliers={
                "educ": 0.0,
                "work": 0.15,
                # vacation_other_multiplier was 0.4 or 0.7
                "other": 0.5,
            },
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-04",
                "end_date": "2021-01-11",
                "prefix": "after-christmas-vacation",
            },
            # hard_lockdown_multipliers were 0.3 or 0.4
            multipliers={
                "educ": 0.0,
                "work": second_jan_week_work_multiplier,
                "other": 0.35,
            },
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-01-12",
                "end_date": "2021-01-31",
                "prefix": "2nd_half_january",
            },
            multipliers={
                "educ": 0.0,
                "work": work_multiplier,
                "other": "other_multiplier",
            },
        ),
        get_soft_lockdown(
            contact_models=contact_models,
            block_info={
                "start_date": "2021-02-01",
                "end_date": "2021-05-01",
                "prefix": "from_feb_onward",
            },
            multipliers=multipliers,
        ),
    ]
    return combine_dictionaries(to_combine)
