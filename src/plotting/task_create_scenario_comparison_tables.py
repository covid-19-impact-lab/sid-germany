import pandas as pd
import pytask

from src.config import BLD
from src.config import FAST_FLAG
from src.config import N_HOUSEHOLDS
from src.config import POPULATION_GERMANY
from src.config import SRC
from src.plotting.plotting import make_name_nice
from src.plotting.task_plot_scenario_comparisons import SCHOOL_SCENARIOS
from src.policies.policy_tools import combine_dictionaries
from src.policies.policy_tools import filter_dictionary
from src.simulation.scenario_config import create_path_to_scenario_outcome_time_series
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios


def _create_table_path(name):
    return BLD / "tables" / f"{name}_table.tex"


def _create_deps(scenarios, groupby):
    available_scenarios = get_available_scenarios(get_named_scenarios())

    py_dependencies = {
        "config.py": SRC / "config.py",
        "scenario_config.py": SRC / "simulation" / "scenario_config.py",
        "plotting.py": SRC / "plotting" / "plotting.py",
    }

    data_dependencies = {}
    scenarios_to_compare = [s for s in scenarios if s in available_scenarios]
    for scenario in scenarios_to_compare:
        outcome = (
            "newly_infected" if groupby is None else f"newly_infected_by_{groupby}"
        )
        data_dependencies[scenario] = create_path_to_scenario_outcome_time_series(
            scenario, outcome
        )
    dependencies = combine_dictionaries([py_dependencies, data_dependencies])
    return dependencies


WORK_RAPID_TEST_SCENARIOS = [
    "spring_without_work_rapid_tests",
    "spring_baseline",
]


_PARAMETRIZATION = [
    (
        _create_deps(SCHOOL_SCENARIOS, "age_group_rki"),
        "2021-04-06" if FAST_FLAG != "debug" else None,
        "5-14",
        "predicted total infections among 5-14 year olds from Easter until {end_date}",
        _create_table_path("student_infections"),
    ),
    (
        _create_deps(WORK_RAPID_TEST_SCENARIOS, None),
        None,
        None,
        "predicted total infections from {start_date} until {end_date}",
        _create_table_path("role_of_work_rapid_tests"),
    ),
]


@pytask.mark.parametrize(
    "depends_on, start_date, group, name, produces", _PARAMETRIZATION
)
def task_create_infections_across_scenarios_table(
    depends_on, start_date, group, name, produces
):
    data_dependencies = filter_dictionary(lambda x: not x.endswith(".py"), depends_on)
    scenario_to_data = {
        scenario: pd.read_pickle(path) for scenario, path in data_dependencies.items()
    }

    if scenario_to_data:
        scenario_infections = _create_table_with_total_infections(
            scenario_to_data=scenario_to_data,
            start_date=start_date,
            group=group,
            name=name,
        )
    else:
        scenario_infections = pd.DataFrame()

    with open(produces, "w") as f:
        f.write(scenario_infections.to_latex())


def _create_table_with_total_infections(scenario_to_data, start_date, group, name):
    scenario_to_infections = pd.Series(dtype=float)
    scenario_to_infections.index.name = "scenario"

    scaling_factor = POPULATION_GERMANY / N_HOUSEHOLDS

    for scenario, data in scenario_to_data.items():
        if start_date is not None:
            data = data.loc[start_date:]
        if group is not None:
            data = data.unstack().swaplevel(axis=1)
            total_infected = int(data[group].sum().mean() * scaling_factor)
        else:
            total_infected = int(scaling_factor * data.sum().mean())
        nice_name = make_name_nice(scenario).replace("\n", " ")
        scenario_to_infections[nice_name] = total_infected

    scenario_to_infections.name = name.format(
        start_date=data.index.min().date(), end_date=data.index.max().date()
    )
    return scenario_to_infections
