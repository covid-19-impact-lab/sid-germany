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
from src.simulation.spec_christmas_scenarios import (
    SCENARIOS,
    CHRISTMAS_MODES,
    CONTACT_TRACING_MULTIPLIERS,
    create_path_to_time_series,
    create_path_to_last_states,
)

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def _create_path_for_effect_of_contact_tracing(scenario, christmas_mode):
    return (
        BLD
        / "simulation"
        / f"effect_of_private_contact_tracing_{christmas_mode}_{scenario}.png"
    )


def _create_parametrization_for_effect_of_private_contact_tracing():
    return [
        (
            scenario,
            christmas_mode,
            {
                ctm: create_path_to_time_series(scenario, christmas_mode, ctm)
                for ctm in CONTACT_TRACING_MULTIPLIERS
            },
            {
                ctm: create_path_to_last_states(scenario, christmas_mode, ctm)
                for ctm in CONTACT_TRACING_MULTIPLIERS
            },
            _create_path_for_effect_of_contact_tracing(scenario, christmas_mode),
        )
        for scenario, christmas_mode in itertools.product(SCENARIOS, CHRISTMAS_MODES)
    ]


@pytask.mark.parametrize(
    "scenario, christmas_mode, paths, depends_on, produces",
    _create_parametrization_for_effect_of_private_contact_tracing(),
)
def task_plot_effect_of_private_contact_tracing(
    scenario, christmas_mode, paths, produces
):
    contact_tracing_scenarios = {
        ctm: dd.read_parquet(path) for ctm, path in paths.items()
    }

    fig, axes = plot_scenarios(contact_tracing_scenarios)
    fig.savefig(produces, dpi=200, bbox_inches="tight", pad_inches=0.5)


def _create_path_for_effect_of_christmas_model(scenario, ctm):
    return (
        BLD
        / "simulation"
        / f"effect_of_christmas_mode_with_{ctm}_contact_tracing_{scenario}.png"
    )


def _create_parametrization_for_effect_of_christmas_mode():
    return [
        (
            scenario,
            ctm,
            {
                cm: create_path_to_time_series(scenario, cm, ctm)
                for cm in CHRISTMAS_MODES
            },
            {
                cm: create_path_to_last_states(scenario, cm, ctm)
                for cm in CHRISTMAS_MODES
            },
            _create_path_for_effect_of_christmas_model(scenario, ctm),
        )
        for scenario, ctm in itertools.product(SCENARIOS, CONTACT_TRACING_MULTIPLIERS)
    ]


@pytask.mark.parametrize(
    "scenario, contact_tracing_multiplier, paths, depends_on, produces",
    _create_parametrization_for_effect_of_christmas_mode(),
)
def task_plot_effect_of_christmas_mode(
    scenario, contact_tracing_multiplier, paths, produces
):
    christmas_mode_scenarios = {
        cm: dd.read_parquet(path) for cm, path in paths.items()
    }

    fig, axes = plot_scenarios(christmas_mode_scenarios)
    for ax in axes.flatten():
        ax.grid(axis="y")
    fig.savefig(produces, dpi=200, bbox_inches="tight", pad_inches=0.5)


def _create_path_for_effect_of_scenario(christmas_mode, ctm):
    return (
        BLD / "simulation" / f"effect_of_optimism_with_{ctm}_contact_tracing_"
        f"and_{christmas_mode}_christmas.png"
    )


def _create_parametrization_for_effect_of_scenario():
    return [
        (
            cm, ctm,
            {
                scenario: create_path_to_time_series(scenario, cm, ctm)
                for scenario in SCENARIOS
            },
            {
                scenario: create_path_to_last_states(scenario, cm, ctm)
                for scenario in SCENARIOS
            },
            _create_path_for_effect_of_scenario(cm, ctm),
        )
        for cm, ctm in itertools.product(CHRISTMAS_MODES, CONTACT_TRACING_MULTIPLIERS)
    ]


@pytask.mark.parametrize(
    "christmas_mode, contact_tracing_multiplier, paths, depends_on, produces",
    _create_parametrization_for_effect_of_scenario(),
)
def task_plot_effect_of_scenario(
    christmas_mode, contact_tracing_multiplier, paths, produces
):
    scenarios = {ctm: dd.read_parquet(path) for ctm, path in paths.items()}

    fig, axes = plot_scenarios(scenarios)
    for ax in axes.flatten():
        ax.grid(axis="y")
    fig.savefig(produces, dpi=200, bbox_inches="tight", pad_inches=0.5)


def plot_scenarios(scenarios):
    name_to_label = {
        None: "Ohne private\nKontaktnachverfolgung",
        0.5: "Mit 50 prozentiger privater\nKontaktnachverfolgung",
        0.1: "Mit 90 prozentiger privater\nKontaktnachverfolgung",
        "full": "Weihnachtsfeiern mit\nwechselnden Personenkreisen",
        "same_group": "Weihnachtsfeiern mit\neinem festen Personenkreis",
        "optimistic": "Optimistisch",
        "pessimistic": "Pessimistisch",
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
                window=7,
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
        top = 350 if outcome == "new_known_case" else 1400
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
