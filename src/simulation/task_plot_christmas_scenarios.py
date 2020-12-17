import dask.dataframe as dd
import matplotlib.pyplot as plt
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
SIMULATIONS = {entry[:2]: entry[2] for entry in simulation_parametrization}


@pytask.mark.depends_on(SIMULATIONS)
@pytask.mark.produces(
    {
        mode: BLD / "simulation" / f"effect_of_private_contact_tracing_{mode}.png"
        for mode in ["full", "same_group", "meet_twice"]
    }
)
def task_plot_effect_of_private_contact_tracing(depends_on, produces):
    for christmas_mode in ["full", "same_group", "meet_twice"]:
        contact_tracing_scenarios = {}
        for (mode, ct_str), path in depends_on.items():
            if mode == christmas_mode:
                df = dd.read_parquet(path / "time_series")
                contact_tracing_scenarios[ct_str] = df

        title = "The Importance of Self-Isolation and Private Contact Tracing"
        fig = plot_scenarios(contact_tracing_scenarios, title=title)
        fig.savefig(produces[christmas_mode])


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
    for ct_mode in [None, 0.5, 0.1]:
        christmas_scenarios = {}
        for (mode, ct_str), path in depends_on.items():
            if ct_str == ct_mode:
                df = dd.read_parquet(path / "time_series")
                christmas_scenarios[mode] = df

        title = "The Importance of Celebrating in Small Circles"
        fig = plot_scenarios(christmas_scenarios, title=title)
        fig.savefig(produces[ct_mode])


def plot_scenarios(scenarios, title):
    name_to_label = {
        None: "Without Private Contact Tracing",
        0.5: "If 50% Follow Private Contact Tracing",
        0.1: "If 90% Follow Private Contact Tracing",
        "full": "Celebrating With Different Households",
        "same_group": "Celebrating in Fixed Groups",
        "meet_twice": "Less Celebrations in Fixed Groups",
    }

    outcome_vars = [
        ("new_known_case", "New Reported Cases"),
        ("newly_infected", "New True Cases"),
    ]
    fig, axes = plt.subplots(
        ncols=len(outcome_vars), figsize=(12, 5), sharey=True, sharex=True
    )

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

    It's smoothed over the time window and scaled as incidence over 100 000.

    """
    time_series = smoothed_outcome_per_hundred_thousand_sim(
        df=df,
        outcome=outcome,
        groupby=None,
        window=window,
        min_periods=min_periods,
        take_logs=False,
    )

    time_series = time_series.compute().reset_index()
    upper_lim = max(ax.get_ylim()[1], time_series[outcome].max())
    sns.lineplot(data=time_series, x="date", y=outcome, ax=ax, label=label, color=color)

    ax.set_ylabel("Smoothed Incidence\nper 100 000 ")
    ax.set_ylim(bottom=0.0, top=upper_lim)

    date_form = DateFormatter("%d.%m")
    ax.xaxis.set_major_formatter(date_form)
