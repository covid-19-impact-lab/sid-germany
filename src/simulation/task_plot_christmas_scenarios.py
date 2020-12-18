import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pytask
import seaborn as sns
from matplotlib.dates import DateFormatter
from sid.colors import get_colors

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
SIMULATIONS = {entry[:2]: entry[3] for entry in simulation_parametrization}


@pytask.mark.depends_on(SIMULATIONS)
@pytask.mark.produces(
    {
        mode: BLD / "simulation" / f"effect_of_private_contact_tracing_{mode}.png"
        for mode in ["full", "same_group", "meet_twice"]
    }
)
def task_plot_effect_of_private_contact_tracing(depends_on, produces):
    named_scenarios = {
        "full": "Weihnachtsfeiern mit wechselnden Haushalten",
        "same_group": "Weihnachtsfeiern im festen Kreis",
        "meet_twice": "Weniger Weihnachtsfeiern im festen Kreis",
    }
    for christmas_mode, name in named_scenarios.items():
        contact_tracing_scenarios = {}
        for (mode, ct_str), path in depends_on.items():
            if mode == christmas_mode:
                df = dd.read_parquet(path)
                contact_tracing_scenarios[ct_str] = df

        title = "Die Bedeutung von privater Kontaktnachverfolgung und Selbstquarantäne"
        fig = plot_scenarios(contact_tracing_scenarios, title=title + "\n" + name)
        fig.savefig(produces[christmas_mode], dpi=200)


@pytask.mark.depends_on(SIMULATIONS)
@pytask.mark.produces(
    {
        ct_mode: BLD
        / "simulation"
        / f"effect_of_christmas_mode_with_{ct_mode}_contact_tracing.png"
        for ct_mode in [None, 0.5, 0.1]
    }
)
def task_plot_effect_of_christmas_mode(depends_on, produces):
    named_scenarios = {
        None: "Ohne private Kontaktnachverfolgung",
        0.5: "Mit 50 prozentiger privater Kontaktnachverfolgung",
        0.1: "Mit 90 prozentiger privater Kontaktnachverfolgung",
    }
    for ct_mode, name in named_scenarios.items():
        christmas_scenarios = {}
        for (mode, ct_str), path in depends_on.items():
            if ct_str == ct_mode:
                df = dd.read_parquet(path)
                christmas_scenarios[mode] = df

        title = "Die Bedeutung der Form der Weihnachtstreffen"
        fig = plot_scenarios(christmas_scenarios, title=title + "\n" + name)
        fig.savefig(produces[ct_mode], dpi=200)


def plot_scenarios(scenarios, title):
    name_to_label = {
        None: "Ohne private\nKontaktnachverfolgung",
        0.5: "Mit 50 prozentiger privater\nKontaktnachverfolgung",
        0.1: "Mit 90 prozentiger privater\nKontaktnachverfolgung",
        "full": "Weihnachtsfeiern mit\nwechselnden Haushalten",
        "same_group": "Weihnachtsfeiern\nim festen Kreis",
        "meet_twice": "Weniger Weihnachtsfeiern\nim festen Kreis",
    }

    outcome_vars = [
        ("new_known_case", "Beobachtete Inzidenz"),
        ("newly_infected", "Tatsächliche Inzidenz"),
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
            )
    fig.autofmt_xdate()
    fig.suptitle(title)

    fig.tight_layout()
    return fig


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
    )

    if isinstance(daily_incidence, dd.Series):
        daily_incidence = daily_incidence.compute()
    weekly_incidence = 7 * daily_incidence
    data = weekly_incidence.reset_index()
    sns.lineplot(data=data, x="date", y=outcome, ax=ax, label=label, color=color)

    ax.set_ylabel("Geglättete wöchentliche \nInzidenz pro 100 000")
    ax.set_xlabel("Datum")
    ax.set_ylim(bottom=0)
    plt.yticks(np.arange(0, weekly_incidence.max(), 100))
    ax.grid(axis="y")

    # only have legend in the left plot
    if outcome == "newly_infected":
        ax.get_legend().remove()

    date_form = DateFormatter("%d.%m")
    ax.xaxis.set_major_formatter(date_form)
