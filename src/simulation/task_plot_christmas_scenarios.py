import itertools

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
<<<<<<< HEAD
=======
from matplotlib.dates import AutoDateLocator
>>>>>>> main
from matplotlib.dates import DateFormatter
from sid.colors import get_colors

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.calculate_moments import smoothed_outcome_per_hundred_thousand_sim
from src.config import BLD
<<<<<<< HEAD
from src.simulation.spec_christmas_scenarios import CHRISTMAS_MODES
from src.simulation.spec_christmas_scenarios import CONTACT_TRACING_MULTIPLIERS
from src.simulation.spec_christmas_scenarios import create_path_to_last_states
from src.simulation.spec_christmas_scenarios import create_path_to_time_series
from src.simulation.spec_christmas_scenarios import SCENARIOS
=======
from src.simulation.task_simulate_christmas_scenarios import (
    create_christmas_parametrization,
)
>>>>>>> main

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


<<<<<<< HEAD
def _named_product(**items):
    """Return each value of the product as a dictionary."""
    for res in itertools.product(*items.values()):
        yield dict(zip(items.keys(), res))


def _create_parametrization(effects, path_func, **cross_product):
    effect_name = list(
        {"scenario", "christmas_mode", "contact_tracing_multiplier"}
        - set(cross_product)
    )[0]
    return [
        (
            {
                eff: create_path_to_time_series(**{**kwargs, **{effect_name: eff}})
                for eff in effects
            },
            {
                eff: create_path_to_last_states(**{**kwargs, **{effect_name: eff}})
                for eff in effects
            },
            path_func(**kwargs),
        )
        for kwargs in _named_product(**cross_product)
    ]


def _create_path_for_effect_of_contact_tracing(scenario, christmas_mode):
    return (
        BLD
        / "simulation"
        / f"effect_of_private_contact_tracing_{christmas_mode}_{scenario}.png"
    )


def _create_path_for_effect_of_christmas_model(scenario, contact_tracing_multiplier):
    return (
        BLD / "simulation" / "effect_of_christmas_mode_with_"
        f"{contact_tracing_multiplier}_contact_tracing_{scenario}.png"
    )


def _create_path_for_effect_of_scenario(christmas_mode, contact_tracing_multiplier):
    return (
        BLD / "simulation" / f"effect_of_optimism_with_{contact_tracing_multiplier}_"
        f"contact_tracing_and_{christmas_mode}_christmas.png"
    )


@pytask.mark.parametrize(
    "paths, depends_on, produces",
    itertools.chain(
        _create_parametrization(
            CONTACT_TRACING_MULTIPLIERS,
            _create_path_for_effect_of_contact_tracing,
            scenario=SCENARIOS,
            christmas_mode=CHRISTMAS_MODES,
        ),
        _create_parametrization(
            CHRISTMAS_MODES,
            _create_path_for_effect_of_christmas_model,
            scenario=SCENARIOS,
            contact_tracing_multiplier=CONTACT_TRACING_MULTIPLIERS,
        ),
        _create_parametrization(
            SCENARIOS,
            _create_path_for_effect_of_scenario,
            christmas_mode=CHRISTMAS_MODES,
            contact_tracing_multiplier=CONTACT_TRACING_MULTIPLIERS,
        ),
    ),
)
def task_plot_one_effect_vs_others(paths, produces):
    scenarios = {scenario: dd.read_parquet(path) for scenario, path in paths.items()}

    fig, axes = plot_scenarios(scenarios)
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
=======
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
        scenarios[label] = dd.read_parquet(depends_on[(christmas_mode, None, optimism)])

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
>>>>>>> main

    outcome_vars = [
        ("newly_infected", "Tatsächliche Inzidenz"),
        ("new_known_case", "Beobachtete Inzidenz"),
    ]
<<<<<<< HEAD
    fig, axes = plt.subplots(ncols=len(outcome_vars), figsize=(8, 4), sharex=True)

    for ax, (outcome, ax_title) in zip(axes, outcome_vars):
        ax.set_title(ax_title)
        colors = get_colors("ordered", 3)
        for color, (name, df) in zip(colors, scenarios.items()):
=======
    fig, axes = plt.subplots(ncols=len(outcome_vars), figsize=(8, 3), sharex=True)

    for ax, (outcome, ax_title) in zip(axes, outcome_vars):
        ax.set_title(ax_title)
        for color, ls, (label, df) in zip(colors, linestyles, scenarios.items()):
>>>>>>> main
            plot_outcome(
                df=df,
                outcome=outcome,
                ax=ax,
<<<<<<< HEAD
                label=name_to_label[name],
                color=color,
                window=7,
=======
                label=label,
                color=color,
                window=7,
                linestyle=ls,
>>>>>>> main
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
<<<<<<< HEAD
            label="Harter Lockdown\nvor Weihnachten",
            color=get_colors("ordered", 3)[0],
        )
=======
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

>>>>>>> main
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
<<<<<<< HEAD
=======
        ax.set_xlim(pd.Timestamp("2020-12-01"), pd.Timestamp("2021-01-10"))
>>>>>>> main

    fig.autofmt_xdate()

    fig.tight_layout()
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.8, -0.5, 1, 0.2), ncol=3)
    axes[0].get_legend().remove()
<<<<<<< HEAD
=======
    loc = AutoDateLocator(minticks=5, maxticks=8)
    for ax in axes:
        ax.xaxis.set_major_locator(loc)
>>>>>>> main
    return fig, axes


def plot_outcome(
    df,
    ax,
    outcome,
    label,
    color,
<<<<<<< HEAD
=======
    linestyle="-",
>>>>>>> main
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
<<<<<<< HEAD
    sns.lineplot(data=data, x="date", y=outcome, ax=ax, label=label, color=color)

    ax.set_ylabel("Geglättete wöchentliche \nNeuinfektionen pro 100.000")
=======
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
>>>>>>> main
    ax.set_xlabel("Datum")

    ax.grid(axis="y")
    ax.set_axisbelow(True)

    date_form = DateFormatter("%d.%m")
    ax.xaxis.set_major_formatter(date_form)
