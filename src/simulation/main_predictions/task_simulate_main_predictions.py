"""Basic prognosis 6 weeks into the future."""
import pandas as pd
import pytask
from sid import get_simulate_func

from src.config import BLD
from src.config import FAST_FLAG
from src.create_initial_states.create_initial_conditions import (
    create_initial_conditions,
)
from src.policies.enacted_policies import get_enacted_policies
from src.policies.policy_tools import shorten_policies
from src.simulation.main_specification import build_main_scenarios
from src.simulation.main_specification import load_simulation_inputs
from src.simulation.main_specification import PREDICT_PATH
from src.simulation.main_specification import SCENARIO_START
from src.simulation.main_specification import SIMULATION_DEPENDENCIES


NESTED_PARAMETRIZATION = build_main_scenarios(PREDICT_PATH)
PARAMETRIZATION = []
for scenario_spec_list in NESTED_PARAMETRIZATION.values():
    PARAMETRIZATION += scenario_spec_list
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
    depends_on,
    produces,
    scenario,  # noqa: U100
    rapid_test_models,
    rapid_test_reaction_models,
    seed,
):
    start_date = pd.Timestamp("2021-03-15")

    if FAST_FLAG in ["debug", "verify"]:
        duration = pd.Timedelta(weeks=8)
    elif FAST_FLAG == "full":
        duration = pd.Timedelta(weeks=12)

    end_date = start_date + duration
    assert end_date > SCENARIO_START, "The scenario start must lie before the end date."

    init_start = start_date - pd.Timedelta(31, unit="D")
    init_end = start_date - pd.Timedelta(1, unit="D")

    virus_shares, simulation_inputs = load_simulation_inputs(
        depends_on,
        init_start,
        end_date,
    )

    initial_conditions = create_initial_conditions(
        start=init_start,
        end=init_end,
        seed=3930,
        reporting_delay=5,
        virus_shares=virus_shares,
        synthetic_data_path=SIMULATION_DEPENDENCIES["initial_states"],
    )

    policies = get_enacted_policies(contact_models=simulation_inputs["contact_models"])
    policies = shorten_policies(policies, init_start, end_date)

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
