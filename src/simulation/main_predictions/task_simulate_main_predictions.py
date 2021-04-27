"""Basic prognosis 6 weeks into the future."""
import pandas as pd
import pytask
from sid import get_simulate_func

from src.config import BLD
from src.config import FAST_FLAG
from src.create_initial_states.create_initial_conditions import (
    create_initial_conditions,
)
from src.policies.combine_policies_over_periods import get_enacted_policies_of_2021
from src.policies.full_policy_blocks import get_lockdown_with_multipliers
from src.policies.policy_tools import combine_dictionaries
from src.simulation.main_specification import build_main_scenarios
from src.simulation.main_specification import load_simulation_inputs
from src.simulation.main_specification import PREDICT_PATH
from src.simulation.main_specification import SCENARIO_START
from src.simulation.main_specification import SIMULATION_DEPENDENCIES


NESTED_PARAMETRIZATION = build_main_scenarios(PREDICT_PATH)
PARAMETRIZATION = [
    spec for seed_list in NESTED_PARAMETRIZATION.values() for spec in seed_list
]
"""Each specification consists of a produces path, the scenario dictioary and a seed"""

if FAST_FLAG == "debug":
    SIMULATION_DEPENDENCIES["initial_states"] = (
        BLD / "data" / "debug_initial_states.parquet"
    )


@pytask.mark.depends_on(SIMULATION_DEPENDENCIES)
@pytask.mark.parametrize(
    "produces, scenario, rapid_test_models, rapid_test_reaction_models, seed",
    PARAMETRIZATION,
)
def task_simulate_main_prediction(
    depends_on, produces, scenario, rapid_test_models, rapid_test_reaction_models, seed
):
    early_start_date = pd.Timestamp("2021-02-15")
    late_start_date = pd.Timestamp("2021-03-13")
    if FAST_FLAG == "debug":
        start_date = late_start_date
    else:
        start_date = early_start_date

    if FAST_FLAG == "debug":
        duration = pd.Timedelta(weeks=4)
    elif FAST_FLAG == "verify":
        duration = pd.Timedelta(weeks=8)
    elif FAST_FLAG == "full":
        duration = pd.Timedelta(weeks=12)

    end_date = start_date + duration
    init_start = start_date - pd.Timedelta(31, unit="D")
    init_end = start_date - pd.Timedelta(1, unit="D")

    scenario_name = produces.parent.name
    test_demand_log_path = (
        produces.parent.parent / "test_demand_logging" / scenario_name
    )
    test_demand_log_path.mkdir(parents=True, exist_ok=True)

    virus_shares, simulation_inputs = load_simulation_inputs(
        depends_on,
        init_start,
        end_date,
        test_demand_log_path=test_demand_log_path,
        extend_ars_dfs=True,
    )

    initial_conditions = create_initial_conditions(
        start=init_start,
        end=init_end,
        seed=3930,
        reporting_delay=5,
        virus_shares=virus_shares,
    )

    policies = _get_prediction_policies(
        contact_models=simulation_inputs["contact_models"],
        scenario=scenario,
        end_date=end_date,
        scenario_name=produces.parent.name,
    )

    simulate = get_simulate_func(
        **simulation_inputs,
        rapid_test_models=rapid_test_models,
        rapid_test_reaction_models=rapid_test_reaction_models,
        contact_policies=policies,
        duration={"start": start_date, "end": end_date},
        initial_conditions=initial_conditions,
        path=produces.parent,
        seed=seed,
        saved_columns={
            "initial_states": ["age_group_rki"],
            "disease_states": ["newly_infected", "infectious", "ever_infected"],
            "time": ["date"],
            "other": [
                "new_known_case",
                "virus_strain",
                "n_has_infected",
                "pending_test",
            ],
        },
    )
    simulate(simulation_inputs["params"])


def _get_prediction_policies(contact_models, scenario, end_date, scenario_name):
    enacted_policies = get_enacted_policies_of_2021(
        contact_models=contact_models,
        scenario_start=SCENARIO_START,
        work_hygiene_multiplier=1.0,
    )
    attend_multiplier = _process_attend_multiplier(
        start_date=SCENARIO_START,
        end_date=end_date,
        # 0.68 was the level between 10th of Jan and carnival
        work_fill_value=scenario.get("work_fill_value", 0.68),
    )
    scenario_policies = get_lockdown_with_multipliers(
        contact_models=contact_models,
        block_info={
            "start_date": SCENARIO_START,
            "end_date": end_date,
            "prefix": scenario_name,
        },
        multipliers={
            "work": {
                "attend_multiplier": attend_multiplier,
                "hygiene_multiplier": scenario.get("work_hygiene_multiplier", 1.0),
            },
            "other": scenario.get("other_multiplier", 0.45),
            "educ": scenario["educ_multiplier"],
        },
        educ_options=scenario.get("educ_options"),
    )
    policies = combine_dictionaries([enacted_policies, scenario_policies])
    return policies


def _process_attend_multiplier(
    start_date,
    end_date,
    attend_multiplier=None,
    work_fill_value=0.68,  # level between 10th of Jan and carnival
):
    dates = pd.date_range(start_date, end_date)
    assert (
        work_fill_value is None or attend_multiplier is None
    ), "work_fill_value may only be supplied if attend_multiplier is None or vice versa"

    if isinstance(attend_multiplier, float):
        return pd.Series(data=attend_multiplier, index=dates)
    elif isinstance(attend_multiplier, pd.Series):
        assert (
            attend_multiplier.index == dates
        ).all(), f"Index is not consecutive from {start_date} to {end_date}"
    elif attend_multiplier is None:
        default_path = BLD / "policies" / "work_multiplier.csv"
        default = pd.read_csv(default_path, parse_dates=["date"], index_col="date")
        expanded = default.reindex(index=dates)
        expanded = expanded.fillna(work_fill_value)
    return expanded
