import itertools

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from matplotlib.dates import DateFormatter
from sid.colors import get_colors

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.calculate_moments import smoothed_outcome_per_hundred_thousand_sim
from src.config import BLD
from src.simulation.task_simulate_christmas_scenarios import (
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
for mode, scenario in itertools.product(
    ["full", "same_group"], ["optimistic", "pessimistic"]
):
    PRODUCTS[f"{mode}_{scenario}"] = (
        BLD / "simulation" / f"effect_of_private_contact_tracing_{mode}_{scenario}.png"
    )


@pytask.mark.depends_on(SIMULATIONS)
@pytask.mark.produces(PRODUCTS)
def task_plot_effect_of_private_contact_tracing(depends_on, produces):
    for scenario in ["optimistic", "pessimistic"]:
        for christmas_mode in ["full", "same_group"]:
            contact_tracing_scenarios = {}
            for (mode, ct_str, sc_str), path in depends_on.items():
                if mode == christmas_mode and sc_str == scenario:
                    df = dd.read_parquet(path)
                    contact_tracing_scenarios[ct_str] = df

            fig, axes = plot_scenarios(contact_tracing_scenarios)
            fig.savefig(
                produces[f"{christmas_mode}_{scenario}"],
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.5,
            )


PRODUCTS = {}
for ct_mode, scenario in itertools.product(
    [None, 0.5, 0.1], ["optimistic", "pessimistic"]
):
    PRODUCTS[f"{ct_mode}_{scenario}"] = (
        BLD
        / "simulation"
        / f"effect_of_christmas_mode_with_{ct_mode}_contact_tracing_{scenario}.png"
    )


@pytask.mark.depends_on(SIMULATIONS)
@pytask.mark.produces(PRODUCTS)
def task_plot_effect_of_christmas_mode(depends_on, produces):
    for scenario in ["optimistic", "pessimistic"]:
        for ct_mode in [None, 0.5, 0.1]:
            christmas_scenarios = {}
            for (mode, ct_str, sc_str), path in depends_on.items():
                if ct_str == ct_mode and sc_str == scenario:
                    df = dd.read_parquet(path)
                    christmas_scenarios[mode] = df

            fig, axes = plot_scenarios(christmas_scenarios)
            for ax in axes.flatten():
                ax.grid(axis="y")
            fig.savefig(
                produces[f"{ct_mode}_{scenario}"],
                dpi=200,
                bbox_inches="tight",
                pad_inches=0.5,
            )


def plot_scenarios(scenarios):
    name_to_label = {
        None: "Ohne private\nKontaktnachverfolgung",
        0.5: "Mit 50 prozentiger privater\nKontaktnachverfolgung",
        0.1: "Mit 90 prozentiger privater\nKontaktnachverfolgung",
        "full": "Weihnachtsfeiern mit\nwechselnden Personenkreisen",
        "same_group": "Weihnachtsfeiern mit\neinem festen Personenkreis",
    }

    outcome_vars = [
        ("newly_infected", "Tatsächliche Inzidenz"),
        ("new_known_case", "Beobachtete Inzidenz"),
    ]
    fig, axes = plt.subplots(ncols=len(outcome_vars), figsize=(8, 4), sharex=True)

    for ax, (outcome, ax_title) in zip(axes, outcome_vars):
        ax.set_title(ax_title)
        colors = get_colors("ordered", 3)
        for color, (name, df) in zip(colors, scenarios.items()):
            plot_outcome(
                df=df,
                outcome=outcome,
                ax=ax,
                label=name_to_label[name],
                color=color,
                window=3,  ### 7,
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
            label="Harter Lockdown\nvor Weihnachten",
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
        top = 500 if outcome == "new_known_case" else 2000  ### 850
        ax.set_ylim(bottom=50, top=top)

    fig.autofmt_xdate()

    fig.tight_layout()
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.8, -0.5, 1, 0.2), ncol=3)
    axes[0].get_legend().remove()
    return fig, axes


def plot_outcome(
    df,
    ax,
    outcome,
    label,
    color,
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
    sns.lineplot(data=data, x="date", y=outcome, ax=ax, label=label, color=color)

    ax.set_ylabel("Geglättete wöchentliche \nNeuinfektionen pro 100 000")
    ax.set_xlabel("Datum")

    ax.grid(axis="y")
    ax.set_axisbelow(True)

    date_form = DateFormatter("%d.%m")
    ax.xaxis.set_major_formatter(date_form)
