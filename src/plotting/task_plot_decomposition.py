"""This module holds the code to compute marginal contributions and shapley values."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import FAST_FLAG
from src.plotting.plotting import format_date_axis
from src.plotting.plotting import ORANGE
from src.plotting.plotting import RED
from src.plotting.plotting import TEAL
from src.simulation.scenario_config import create_path_to_scenario_outcome_time_series
from src.simulation.scenario_config import get_available_scenarios
from src.simulation.scenario_config import get_named_scenarios


_SCENARIO_TO_MEMBERS = {
    "spring_baseline": frozenset(["rapid_tests", "vaccinations", "seasonality"]),
    "spring_without_rapid_tests_and_no_vaccinations": frozenset(["seasonality"]),
    "spring_without_vaccinations_without_seasonality": frozenset(["rapid_tests"]),
    "spring_without_rapid_tests_without_seasonality": frozenset(["vaccinations"]),
    "spring_without_rapid_tests": frozenset(["vaccinations", "seasonality"]),
    "spring_without_vaccines": frozenset(["rapid_tests", "seasonality"]),
    "spring_without_seasonality": frozenset(["rapid_tests", "vaccinations"]),
    "spring_no_effects": frozenset([]),
}

_SCENARIOS = list(_SCENARIO_TO_MEMBERS)

_AVAILABLE_SCENARIOS = get_available_scenarios(get_named_scenarios())

_ARE_ALL_SCENARIOS_AVAILABLE = all(i in _AVAILABLE_SCENARIOS for i in _SCENARIOS)

_DEPENDS_ON = {
    name: create_path_to_scenario_outcome_time_series(name, "newly_infected")
    for name in _SCENARIOS
}

_MATPLOTLIB_RC_CONTEXT = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
}

_ORDERED_CHANNELS = ["Rapid Tests", "Seasonality", "Vaccinations"]


@pytask.mark.skipif(
    not _ARE_ALL_SCENARIOS_AVAILABLE, reason="required scenarios are not available"
)
@pytask.mark.depends_on(_DEPENDS_ON)
@pytask.mark.produces(
    {
        "bar_plot": BLD / "figures" / f"{FAST_FLAG}_decomposition_bar.pdf",
        "area_plot": BLD / "figures" / f"{FAST_FLAG}_decomposition_area.pdf",
    }
)
def task_plot_decomposition_of_infection_channels_in_spring(depends_on, produces):
    scenarios = {name: pd.read_pickle(path) for name, path in depends_on.items()}

    df = pd.DataFrame()
    for name, s in scenarios.items():
        cumulative_outcomes = s.mean(axis=1)
        df[name] = cumulative_outcomes

    fig = _create_bar_plot(df)
    fig.savefig(produces["bar_plot"])

    fig = _create_area_plot(df)
    fig.savefig(produces["area_plot"])


def _create_bar_plot(df):
    ratios = _compute_ratios_based_on_shapley_values(df.sum())
    ratios = ratios.rename(
        index=lambda x: x.replace("shapley_value_", "").replace("_", " ").title()
    ).reindex(index=_ORDERED_CHANNELS)

    with plt.rc_context(_MATPLOTLIB_RC_CONTEXT):
        fig, ax = plt.subplots()

        ratios.plot(kind="barh", ax=ax, color=[RED, ORANGE, TEAL], alpha=0.6)

        ax.set_xlabel("Contribution to Reduction")

    return fig


def _create_area_plot(df):
    ratios = df.cumsum().apply(_compute_ratios_based_on_shapley_values, axis=1)
    prevented_infections = df["spring_no_effects"] - df["spring_baseline"]

    # Clipping is necessary for the area plot and only small numbers in the beginning
    # are clipped which do not change the results.
    prevented_infections_by_channel = (
        (ratios.multiply(prevented_infections.cumsum(), axis=0).diff().clip(0))
        .rename(
            columns=lambda x: x.replace("shapley_value_", "").replace("_", " ").title()
        )
        .reindex(columns=_ORDERED_CHANNELS)
    )

    with plt.rc_context(_MATPLOTLIB_RC_CONTEXT):
        fig, ax = plt.subplots()
        prevented_infections_by_channel.plot(
            kind="area", ax=ax, color=[RED, ORANGE, TEAL], alpha=0.6
        )

        ax = format_date_axis(ax)

        ax.set_ylabel("smoothed weekly incidence")
        ax.set_xlabel(None)
        ax.grid(axis="y")
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
        )

        x, y, width, height = 0, -0.3, 1, 0.15
        ax.legend(loc="upper center", bbox_to_anchor=(x, y, width, height), ncol=3)

        fig.tight_layout()

    return fig


def _compute_ratios_based_on_shapley_values(s):
    df = s.to_frame(name="cumulative_infections")
    df["members"] = df.index.map(_SCENARIO_TO_MEMBERS.get)

    df["payoff"] = (
        df["cumulative_infections"]
        - df.loc["spring_no_effects", "cumulative_infections"]
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
