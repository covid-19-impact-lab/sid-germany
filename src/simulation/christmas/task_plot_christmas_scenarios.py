import itertools

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from matplotlib.dates import AutoDateLocator
from matplotlib.dates import DateFormatter
from sid.colors import get_colors

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.calculate_moments import smoothed_outcome_per_hundred_thousand_sim
from src.config import BLD
from src.simulation.christmas.task_simulate_christmas_scenarios import (
    create_christmas_parametrization,
)

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


simulation_parametrization = create_christmas_parametrization()
SIMULATIONS = {entry[:3]: entry[4] for entry in simulation_parametrization}

PRODUCTS = {}
for mode, optimism in itertools.product(
    ["full", "same_group"], ["optimistic", "pessimistic"]
):
    PRODUCTS[f"{mode}_{optimism}"] = (
        BLD / "simulation" / f"effect_of_private_contact_tracing_{mode}_{optimism}.png"
    )

NAME_TO_LABEL = {
    None: "Ohne private\nKontaktnachverfolgung",
    0.5: "Mit 50 prozentiger privater\nKontaktnachverfolgung",
    0.1: "Mit 90 prozentiger privater\nKontaktnachverfolgung",
    "full": "Weihnachtsfeiern mit\nwechselnden Personenkreisen",
    "same_group": "Weihnachtsfeiern mit\neinem festen Personenkreis",
    "optimistic": "Optimistisch",
    "pessimistic": "Pessimistisch",
}


@pytask.mark.skip
@pytask.mark.depends_on(SIMULATIONS)
@pytask.mark.produces(PRODUCTS)
def task_plot_effect_of_private_contact_tracing(depends_on, produces):
    for optimism in ["optimistic", "pessimistic"]:
        for christmas_mode in ["full", "same_group"]:
            contact_tracing_scenarios = {}
            for (mode, ct_str, optim_str), path in depends_on.items():
                if mode == christmas_mode and optim_str == optimism:
                    df = dd.read_parquet(path)
                    label = NAME_TO_LABEL[ct_str]
                    contact_tracing_scenarios[label] = df

            fig, axes = plot_scenarios(contact_tracing_scenarios)
            fig.savefig(
                produces[f"{christmas_mode}_{optimism}"],
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.5,
            )


PRODUCTS = {}
for ct_mode, optimism in itertools.product(
    [None, 0.5, 0.1], ["optimistic", "pessimistic"]
):
    PRODUCTS[f"{ct_mode}_{optimism}"] = (
        BLD
        / "simulation"
        / f"effect_of_christmas_mode_with_{ct_mode}_contact_tracing_{optimism}.png"
    )


@pytask.mark.skip
@pytask.mark.depends_on(SIMULATIONS)
@pytask.mark.produces(PRODUCTS)
def task_plot_effect_of_christmas_mode(depends_on, produces):
    for optimism in ["optimistic", "pessimistic"]:
        for ct_mode in [None, 0.5, 0.1]:
            christmas_scenarios = {}
            for (mode, ct_str, optimism_str), path in depends_on.items():
                if ct_str == ct_mode and optimism_str == optimism:
                    df = dd.read_parquet(path)
                    christmas_scenarios[NAME_TO_LABEL[mode]] = df

            fig, axes = plot_scenarios(christmas_scenarios)
            for ax in axes.flatten():
                ax.grid(axis="y")
            fig.savefig(
                produces[f"{ct_mode}_{optimism}"],
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.5,
            )


PRODUCTS = {}
for ct_mode, christmas_mode in itertools.product(
    [None, 0.5, 0.1], ["full", "same_group"]
):
    PRODUCTS[f"{ct_mode}_{christmas_mode}"] = (
        BLD / "simulation" / f"effect_of_optimism_with_{ct_mode}_contact_tracing_"
        f"and_{christmas_mode}_christmas.png"
    )


@pytask.mark.skip
@pytask.mark.depends_on(SIMULATIONS)
@pytask.mark.produces(PRODUCTS)
def task_plot_effect_of_optimism(depends_on, produces):
    for christmas_mode in ["full", "same_group"]:
        for ct_mode in [None, 0.5, 0.1]:
            scenarios = {}
            for (mode, ct_str, optimism_str), path in depends_on.items():
                if ct_str == ct_mode and christmas_mode == mode:
                    df = dd.read_parquet(path)
                    scenarios[NAME_TO_LABEL[optimism_str]] = df

            fig, axes = plot_scenarios(scenarios)
            for ax in axes.flatten():
                ax.grid(axis="y")
            fig.savefig(
                produces[f"{ct_mode}_{christmas_mode}"],
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.5,
            )


@pytask.mark.skip
@pytask.mark.depends_on(SIMULATIONS)
@pytask.mark.produces(BLD / "simulation" / "figure2.png")
def task_plot_figure_two(depends_on, produces):
    # christmas_mode, contact_tracing, optimism
    label_dict = {
        "full": "Weihnachtsfeiern mit\nwechselnden Personenkreisen",
        "same_group": "Weihnachtsfeiern mit\neinem festen Personenkreis",
        "optimistic": "optimistisches Szenario",
        "pessimistic": "pessimistisches Szenario",
    }
    scenario_tuples = [
        ("full", "optimistic"),
        ("same_group", "optimistic"),
        ("full", "pessimistic"),
        ("same_group", "pessimistic"),
    ]
    scenarios = {}
    for (christmas_mode, optimism) in scenario_tuples:
        label = f"{label_dict[christmas_mode]},\n{label_dict[optimism]}"
        scenarios[label] = dd.read_parquet(depends_on[(optimism, christmas_mode, None)])

    colors = get_colors("ordered", 3)[:2] * 2  # color identifies christmas mode
    linestyles = ["-", "-", "--", "--"]  # line style gives optimism

    fig, axes = plot_scenarios(
        scenarios,
        colors=colors,
        linestyles=linestyles,
    )
    for ax in axes.flatten():
        ax.grid(axis="y")

    # drop shaded area labels
    handles, labels = axes[1].get_legend_handles_labels()
    handles = handles[:-2]
    labels = labels[:-2]
    axes[1].legend(
        handles, labels, loc="upper center", bbox_to_anchor=(-0.8, -0.5, 1, 0.2), ncol=3
    )

    fig.savefig(
        produces,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.5,
    )


DEFAULT_COLORS = get_colors("ordered", 3)


def plot_scenarios(scenarios, colors=DEFAULT_COLORS, linestyles=None):
    if linestyles is None:
        linestyles = ["-"] * len(scenarios)

    outcome_vars = [
        ("newly_infected", "Tatsächliche Inzidenz"),
        ("new_known_case", "Beobachtete Inzidenz"),
    ]
    fig, axes = plt.subplots(ncols=len(outcome_vars), figsize=(8, 3), sharex=True)

    for ax, (outcome, ax_title) in zip(axes, outcome_vars):
        ax.set_title(ax_title)
        for color, ls, (label, df) in zip(colors, linestyles, scenarios.items()):
            plot_outcome(
                df=df,
                outcome=outcome,
                ax=ax,
                label=label,
                color=color,
                window=7,
                linestyle=ls,
            )

        ax.fill_between(
            x=[pd.Timestamp("2020-12-24"), pd.Timestamp("2020-12-26")],
            y1=0,
            y2=2000,
            alpha=0.2,
            label="Weihnachten",
            color=get_colors("ordered", 3)[2],
        )
        ax.fill_between(
            x=[pd.Timestamp("2020-12-16"), pd.Timestamp("2020-12-24")],
            y1=0,
            y2=2000,
            alpha=0.2,
            label="Harter Lockdown",
            color=get_colors("ordered", 3)[0],
        )
        ax.fill_between(
            x=[pd.Timestamp("2020-12-26"), pd.Timestamp("2021-01-10")],
            y1=0,
            y2=2000,
            alpha=0.2,
            color=get_colors("ordered", 3)[0],
        )

        if outcome == "new_known_case":
            rki_data = pd.read_pickle(
                BLD / "data" / "processed_time_series" / "rki.pkl"
            )
            rki_incidence = 7 * smoothed_outcome_per_hundred_thousand_rki(
                rki_data, "newly_infected", take_logs=False
            )
            rki_incidence = rki_incidence["2020-12-01":].reset_index()
            sns.lineplot(
                data=rki_incidence,
                x="date",
                y="newly_infected",
                label="RKI Fallzahlen",
                color="k",
                ax=ax,
            )
        top = 350 if outcome == "new_known_case" else 1400
        ax.set_ylim(bottom=50, top=top)
        ax.set_xlim(pd.Timestamp("2020-12-01"), pd.Timestamp("2021-01-10"))

    fig.autofmt_xdate()

    fig.tight_layout()
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.8, -0.5, 1, 0.2), ncol=3)
    legend_to_remove = axes[0].get_legend()
    if legend_to_remove is not None:
        legend_to_remove.remove()
    loc = AutoDateLocator(minticks=5, maxticks=8)
    for ax in axes:
        ax.xaxis.set_major_locator(loc)
    return fig, axes


def plot_outcome(
    df,
    ax,
    outcome,
    label,
    color,
    linestyle="-",
    window=7,
    min_periods=1,
):
    """Plot an outcome variable over time.

    It's smoothed over the time window and scaled as weekly incidence over 100 000.

    """
    daily_incidence = smoothed_outcome_per_hundred_thousand_sim(
        df=df,
        outcome=outcome,
        groupby=None,
        window=window,
        min_periods=min_periods,
        take_logs=False,
        center=False,
    )

    if isinstance(daily_incidence, dd.Series):
        daily_incidence = daily_incidence.compute()
    weekly_incidence = 7 * daily_incidence
    data = weekly_incidence.reset_index()
    sns.lineplot(
        data=data,
        x="date",
        y=outcome,
        ax=ax,
        label=label,
        color=color,
        linestyle=linestyle,
    )

    ax.set_ylabel("Geglättete wöchentliche \nNeuinfektionen pro 100 000")
    ax.set_xlabel("Datum")

    ax.grid(axis="y")
    ax.set_axisbelow(True)

    date_form = DateFormatter("%d.%m")
    ax.xaxis.set_major_formatter(date_form)
