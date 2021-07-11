"""This module holds the code to compute marginal contributions and shapley values."""
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import FAST_FLAG
from src.config import PLOT_SIZE
from src.plotting.plotting import BLUE
from src.plotting.plotting import format_date_axis
from src.plotting.plotting import GREEN
from src.plotting.plotting import ORANGE
from src.plotting.plotting import RED
from src.plotting.plotting import TEAL
from src.plotting.plotting import YELLOW
from src.simulation.scenario_config import create_path_to_scenario_outcome_time_series
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios


_CHANNEL_SCENARIOS_TO_MEMBERS = {
    "spring_baseline": frozenset(["rapid_tests", "vaccinations", "seasonality"]),
    "spring_without_rapid_tests_and_no_vaccinations": frozenset(["seasonality"]),
    "spring_without_vaccinations_without_seasonality": frozenset(["rapid_tests"]),
    "spring_without_rapid_tests_without_seasonality": frozenset(["vaccinations"]),
    "spring_without_rapid_tests": frozenset(["vaccinations", "seasonality"]),
    "spring_without_vaccines": frozenset(["rapid_tests", "seasonality"]),
    "spring_without_seasonality": frozenset(["rapid_tests", "vaccinations"]),
    "spring_no_effects": frozenset([]),
}

_RAPID_TEST_SCENARIOS_TO_MEMBERS = {
    "spring_baseline": frozenset(["private", "school", "work"]),
    "spring_without_school_and_work_rapid_tests": frozenset(["private"]),
    "spring_without_school_and_private_rapid_tests": frozenset(["work"]),
    "spring_without_work_and_private_rapid_tests": frozenset(["school"]),
    "spring_without_work_rapid_tests": frozenset(["school", "private"]),
    "spring_without_school_rapid_tests": frozenset(["work", "private"]),
    "spring_without_private_rapid_tests": frozenset(["school", "work"]),
    "spring_without_rapid_tests": frozenset([]),
}

_AVAILABLE_SCENARIOS = get_available_scenarios(get_named_scenarios())

_MATPLOTLIB_RC_CONTEXT = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
}

_ORDERED_CHANNELS = ["Rapid Tests", "Seasonality", "Vaccinations"]
_ORDERED_RAPID_TEST_CHANNELS = ["Private", "Work", "School"]

_ALL_CHANNEL_SCENARIOS_AVAILABLE = all(
    i in _AVAILABLE_SCENARIOS for i in _CHANNEL_SCENARIOS_TO_MEMBERS
)
_ALL_RAPID_TEST_SCENARIOS_AVAILABLE = all(
    i in _AVAILABLE_SCENARIOS for i in _RAPID_TEST_SCENARIOS_TO_MEMBERS
)


@pytask.mark.skipif(
    not _ALL_CHANNEL_SCENARIOS_AVAILABLE,
    reason="required scenarios are not available",
)
@pytask.mark.depends_on(
    {
        name: create_path_to_scenario_outcome_time_series(name, "newly_infected")
        for name in _CHANNEL_SCENARIOS_TO_MEMBERS
    }
)
@pytask.mark.produces(
    {
        "bar_plot": BLD / "figures" / f"{FAST_FLAG}_decomposition_channels_bar.pdf",
        "area_plot": BLD / "figures" / f"{FAST_FLAG}_decomposition_channels_area.pdf",
    }
)
def task_plot_decomposition_of_infection_channels_in_spring(depends_on, produces):
    scenarios = {name: pd.read_pickle(path) for name, path in depends_on.items()}

    df = pd.DataFrame()
    for name, s in scenarios.items():
        cumulative_outcomes = s.mean(axis=1)
        df[name] = cumulative_outcomes

    fig = _create_bar_plot(
        df,
        scenario_to_members=_CHANNEL_SCENARIOS_TO_MEMBERS,
        no_effects_scenario="spring_no_effects",
        ordering=_ORDERED_CHANNELS,
        color=[RED, ORANGE, TEAL],
    )
    fig.savefig(produces["bar_plot"])
    plt.close()

    fig, ax = _create_area_plot(
        df,
        scenario_to_members=_CHANNEL_SCENARIOS_TO_MEMBERS,
        no_effects_scenario="spring_no_effects",
        ordering=_ORDERED_CHANNELS,
        color=[RED, ORANGE, TEAL],
    )
    ax.set_xlim(pd.Timestamp("2021-01-15"), None)
    fig.savefig(produces["area_plot"])
    plt.close()


@pytask.mark.skipif(
    not _ALL_RAPID_TEST_SCENARIOS_AVAILABLE,
    reason="required scenarios are not available",
)
@pytask.mark.depends_on(
    {
        name: create_path_to_scenario_outcome_time_series(name, "newly_infected")
        for name in _RAPID_TEST_SCENARIOS_TO_MEMBERS
    }
)
@pytask.mark.produces(
    {
        "bar_plot": BLD / "figures" / f"{FAST_FLAG}_decomposition_rapid_tests_bar.pdf",
        "area_plot": BLD
        / "figures"
        / f"{FAST_FLAG}_decomposition_rapid_tests_area.pdf",
    }
)
def task_plot_decomposition_of_rapid_tests_in_spring(depends_on, produces):
    scenarios = {name: pd.read_pickle(path) for name, path in depends_on.items()}

    df = pd.DataFrame()
    for name, s in scenarios.items():
        cumulative_outcomes = s.mean(axis=1)
        df[name] = cumulative_outcomes

    fig = _create_bar_plot(
        df,
        scenario_to_members=_RAPID_TEST_SCENARIOS_TO_MEMBERS,
        no_effects_scenario="spring_without_rapid_tests",
        ordering=_ORDERED_RAPID_TEST_CHANNELS,
        color=[BLUE, GREEN, YELLOW],
    )
    fig.savefig(produces["bar_plot"])
    plt.close()

    fig, ax = _create_area_plot(
        df,
        scenario_to_members=_RAPID_TEST_SCENARIOS_TO_MEMBERS,
        no_effects_scenario="spring_without_rapid_tests",
        ordering=_ORDERED_RAPID_TEST_CHANNELS,
        color=[BLUE, GREEN, YELLOW],
    )
    ax.set_xlim(pd.Timestamp("2021-01-15"), None)
    fig.savefig(produces["area_plot"])
    plt.close()


def _create_bar_plot(df, scenario_to_members, no_effects_scenario, ordering, color):
    ratios = _compute_ratios_based_on_shapley_values(
        df.sum(),
        scenario_to_members=scenario_to_members,
        no_effects_scenario=no_effects_scenario,
    )
    ratios = ratios.rename(
        index=lambda x: x.replace("shapley_value_", "").replace("_", " ").title()
    ).reindex(index=ordering)

    with plt.rc_context(_MATPLOTLIB_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=PLOT_SIZE)

        ratios.plot(kind="barh", ax=ax, color=color, alpha=0.6)

        ax.set_xlabel("Contribution to Reduction")
    fig.tight_layout()
    return fig


def _create_area_plot(df, scenario_to_members, no_effects_scenario, ordering, color):
    ratios = df.cumsum().apply(
        _compute_ratios_based_on_shapley_values,
        scenario_to_members=scenario_to_members,
        no_effects_scenario=no_effects_scenario,
        axis=1,
    )
    ratios.columns = [
        x.replace("shapley_value_", "").replace("_", " ").title()
        for x in ratios.columns
    ]

    prevented_infections = df[no_effects_scenario] - df["spring_baseline"]

    # Clipping is necessary for the area plot and only small numbers in the beginning
    # are clipped which do not change the results.
    prevented_infections_by_channel = (
        ratios.multiply(prevented_infections.cumsum(), axis=0).diff().clip(0)
    ).reindex(columns=ordering)
    prevented_infections_by_channel.columns = [
        x + f" ({ratios.iloc[-1][x]:.0%})"
        for x in prevented_infections_by_channel.columns
    ]

    with plt.rc_context(_MATPLOTLIB_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=PLOT_SIZE)
        prevented_infections_by_channel.plot(kind="area", ax=ax, color=color, alpha=0.6)

        ax = format_date_axis(ax)

        ax.set_ylabel(
            "contribution to difference in new cases per million inhabitants per day"
        )
        ax.set_xlabel(None)
        ax.grid(axis="y")
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
        )

        x, y, width, height = 0, -0.3, 1, 0.15
        ax.legend(loc="upper center", bbox_to_anchor=(x, y, width, height), ncol=3)

        fig.tight_layout()

    return fig, ax


def _compute_ratios_based_on_shapley_values(
    s, scenario_to_members: Dict[str, frozenset], no_effects_scenario: str
):
    df = s.to_frame(name="cumulative_infections")
    df["members"] = df.index.map(scenario_to_members.get)

    df["payoff"] = (
        df["cumulative_infections"]
        - df.loc[no_effects_scenario, "cumulative_infections"]
    ) * -1
    df = df.reset_index()
    shapley_values = _compute_shapley_values(df)
    ratios = shapley_values / shapley_values.sum()
    return ratios


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
