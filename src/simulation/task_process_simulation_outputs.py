import pandas as pd
import pytask

from src.calculate_moments import aggregate_and_smooth_period_outcome_sim
from src.config import SRC
from src.simulation.load_simulation_inputs import create_period_outputs
from src.simulation.scenario_config import create_path_to_period_outputs_of_simulation
from src.simulation.scenario_config import create_path_to_scenario_outcome_time_series
from src.simulation.scenario_config import create_path_to_share_known_cases_of_scenario
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios


_MODULE_DEPENDENCIES = {
    "calculate_moments.py": SRC / "calculate_moments.py",
    "load_simulation_inputs.py": SRC / "simulation" / "load_simulation_inputs.py",
    "scenario_config.py": SRC / "simulation" / "scenario_config.py",
}


def _create_create_weekly_incidence_parametrization():
    named_scenarios = get_named_scenarios()
    parametrization = []
    period_output_keys = create_period_outputs().keys()
    for name, specs in named_scenarios.items():
        depends_on = {}
        for seed in range(specs["n_seeds"]):
            depends_on[seed] = create_path_to_period_outputs_of_simulation(name, seed)

        # this handles the case of 0 seeds, i.e. skipped scenarios
        if depends_on:
            produces = {
                entry: create_path_to_scenario_outcome_time_series(name, entry)
                for entry in period_output_keys
            }

            parametrization.append((depends_on, produces))

    return "depends_on, produces", parametrization


_SIGNATURE, _PARAMETRIZATION = _create_create_weekly_incidence_parametrization()


@pytask.mark.depends_on(_MODULE_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_create_weekly_outcome_for_scenario(depends_on, produces):
    seed_keys = [seed for seed in depends_on if isinstance(seed, int)]
    results = {str(seed): pd.read_pickle(depends_on[seed]) for seed in seed_keys}
    for entry, path in produces.items():
        outcome_and_groupby = entry.split("_by_")
        if len(outcome_and_groupby) == 1:
            outcome, groupby = outcome_and_groupby[0], None
        else:
            outcome, groupby = outcome_and_groupby
        out = pd.DataFrame()
        for seed, res in results.items():
            daily_incidence = aggregate_and_smooth_period_outcome_sim(
                res, outcome=entry, groupby=groupby, take_logs=False
            )
            if groupby is not None:
                # ensure that the index is complete
                dates = daily_incidence.index.levels[0].unique()
                groups = daily_incidence.index.levels[1].unique()
                full_index = pd.MultiIndex.from_product([dates, groups])
                daily_incidence = daily_incidence.reindex(full_index).fillna(0)

            if outcome == "ever_vaccinated":
                # share who are vaccinated already
                out[seed] = daily_incidence / 100_000
            else:
                # weekly incidence
                out[seed] = daily_incidence * 7
        out.to_pickle(path)


def _create_scenario_share_known_cases_parametrization():
    """Create the parametrization for the share known cases."""
    named_scenarios = get_named_scenarios()
    available_scenarios = get_available_scenarios(named_scenarios)
    parametrization = []

    for scenario_name in available_scenarios:
        depends_on = {}
        for outcome in ["currently_infected", "knows_currently_infected"]:
            depends_on[outcome] = create_path_to_scenario_outcome_time_series(
                scenario_name, f"{outcome}_by_age_group_rki"
            )
        produces = create_path_to_share_known_cases_of_scenario(scenario_name)
        parametrization.append((depends_on, produces))
    return "depends_on, produces", parametrization


_SIGNATURE, _PARAMETRIZATION = _create_scenario_share_known_cases_parametrization()


@pytask.mark.depends_on(_MODULE_DEPENDENCIES)
@pytask.mark.parametrize(_SIGNATURE, _PARAMETRIZATION)
def task_create_share_known_cases(depends_on, produces):
    knows_currently_infected = pd.read_pickle(depends_on["knows_currently_infected"])
    currently_infected = pd.read_pickle(depends_on["currently_infected"])
    share_known_cases = knows_currently_infected / currently_infected
    share_known_cases["mean"] = share_known_cases.mean(axis=1)
    assert not share_known_cases.index.duplicated().any()
    assert share_known_cases.notnull().all().all()
    share_known_cases.to_pickle(produces)
