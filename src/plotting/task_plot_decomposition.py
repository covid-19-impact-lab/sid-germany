"""This module holds the code to compute marginal contributions and shapley values."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import FAST_FLAG
from src.simulation.scenario_config import create_path_to_scenario_outcome_time_series
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios


_SCENARIOS = [
    "spring_baseline",
    "spring_without_rapid_tests_and_no_vaccinations",
    "spring_without_vaccinations_without_seasonality",
    "spring_without_rapid_tests_without_seasonality",
    "spring_without_rapid_tests",
    "spring_without_vaccines",
    "spring_without_seasonality",
    "spring_no_effects",
]


_AVAILABLE_SCENARIOS = get_available_scenarios(get_named_scenarios())


_ARE_ALL_SCENARIOS_AVAILABLE = all(i in _AVAILABLE_SCENARIOS for i in _SCENARIOS)


_DEPENDS_ON = {
    name: create_path_to_scenario_outcome_time_series(name, "newly_infected")
    for name in _SCENARIOS
}


@pytask.mark.skipif(
    not _ARE_ALL_SCENARIOS_AVAILABLE, reason="required scenarios are not available"
)
@pytask.mark.depends_on(_DEPENDS_ON)
@pytask.mark.produces(BLD / "figures" / f"{FAST_FLAG}_decomposition.pdf")
def task_plot_decomposition(depends_on, produces):
    scenarios = {name: pd.read_pickle(path) for name, path in depends_on.items()}

    df = {"name": [], "payoff": []}
    for name, df_ in scenarios.items():
        df["name"].append(name)
        df["payoff"].append((df_.mean(axis=1)).sum())
    df = pd.DataFrame(df)

    df["members"] = [
        frozenset(["rapid_tests", "vaccinations", "seasonality"]),
        frozenset(["seasonality"]),
        frozenset(["rapid_tests"]),
        frozenset(["vaccinations"]),
        frozenset(["vaccinations", "seasonality"]),
        frozenset(["rapid_tests", "seasonality"]),
        frozenset(["rapid_tests", "vaccinations"]),
        frozenset([]),
    ]

    df["payoff"] = df["payoff"] - df["payoff"].max()

    shapley_values = _compute_shapley_values(df)

    ratios = (shapley_values / shapley_values.sum()) * 100
    ratios = ratios.rename(
        index=lambda x: x.replace("shapley_value_", "").replace("_", " ").title()
    ).sort_values()
    ratios.index.name = None

    fig, ax = plt.subplots()

    ratios.plot(kind="barh", ax=ax)

    ax.set_ylabel("Policy")
    ax.set_xlabel("Contribution to Reduction (in %)")

    plt.tight_layout()

    plt.savefig(produces)


def _compute_shapley_values(coalitions):
    """Compute Shapley values.

    Parameters
    ----------
    coalitions : pandas.DataFrame
        A DataFrame where each row contains one coalition and an associated payoff.

        - ``"members"`` contains an iterable with all the members participating in the
          specific coalition.
        - ``"payoff"`` contains the value associated with the coalition.

    Returns
    -------
    shapley_values: pandas.Series
        The Shapley values associated with each of the members found in all coalitions.

    """
    coalitions_w_marginal_effects = _compute_marginal_effects(coalitions)
    shapley_values = _compute_shapley_values_from_marginal_effects(
        coalitions_w_marginal_effects
    )

    return shapley_values


def _compute_marginal_effects(coalitions):
    coalitions["coalition"] = range(len(coalitions))
    coalitions["members"] = coalitions["members"].map(frozenset)
    all_members = frozenset().union(*coalitions["members"].tolist())
    set_to_code = dict(zip(coalitions["members"], coalitions["coalition"]))

    for member in all_members:
        coalitions[f"marginal_contribution_{member}"] = np.nan

    for index, row in coalitions.iterrows():
        coalition_members = row.members
        payoff = row.payoff
        for member in coalition_members:
            coalition_wo_member = coalition_members - frozenset([member])
            other_index = set_to_code[coalition_wo_member]
            other_payoff = coalitions.loc[other_index, "payoff"]
            coalitions.loc[index, f"marginal_contribution_{member}"] = (
                payoff - other_payoff
            )

    return coalitions


def _compute_shapley_values_from_marginal_effects(coalitions):
    coalitions["coalition_size"] = coalitions["members"].map(len)
    all_members = frozenset().union(*coalitions["members"].tolist())

    shapley_values = (
        coalitions.groupby("coalition_size")[
            [f"marginal_contribution_{member}" for member in all_members]
        ]
        .mean()
        .mean()
    ).rename(index=lambda x: x.replace("marginal_contribution", "shapley_value"))

    return shapley_values
