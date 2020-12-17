import dask.dataframe as dd
import matplotlib.pyplot as plt
import pytask
import seaborn as sns
from sid.colors import get_colors

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_sim
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


def _create_plot_christmas_simulation_parametrization():
    plot_parametrization = []
    simulation_parametrization = create_christmas_parametrization()
    # christmas_model, contact_tracing_multiplier, dir_path, last_states_path
    plot_specification = [
        ("True Infections", {"outcome": "newly_infected"}),
    ]
    for _, _, depends_on, _ in simulation_parametrization:
        for title, specs in plot_specification:
            entry = (
                depends_on,
                title,
                specs,
                depends_on / f"{title.lower().replace(' ', '_')}.png",
            )
            plot_parametrization.append(entry)
    return plot_parametrization


PARAMETRIZATION = _create_plot_christmas_simulation_parametrization()


@pytask.mark.parametrize("depends_on, title, plot_specs, produces", PARAMETRIZATION)
def task_plot_christmas_scenario(depends_on, title, plot_specs, produces):
    df = dd.read_parquet(depends_on / "time_series")
    fig, ax = plot_outcome(df, title, **plot_specs)
    fig.savefig(produces)


def plot_outcome(
    df, title, outcome, groupby=None, window=7, min_periods=1, fig=None, ax=None
):
    """Plot an outcome variable over time.

    It's smoothed over the time window and scaled as incidence over 100 000.

    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    time_series = smoothed_outcome_per_hundred_thousand_sim(
        df=df,
        outcome=outcome,
        groupby=groupby,
        window=window,
        min_periods=min_periods,
        take_logs=False,
    )

    if groupby:
        time_series.name = outcome
        number = len(df[groupby].dtype.categories)
        palette = "ordered" if df[groupby].dtype.ordered else "categorical"
        colors = get_colors(palette=palette, number=number)
    else:
        time_series = time_series.compute()
        colors = get_colors(palette="categorical", number=1)

    sns.color_palette(colors)

    time_series = time_series.reset_index()
    sns.lineplot(data=time_series, x="date", y=outcome, hue=groupby)

    ax.set_ylabel("Smoothed Incidence\nper 100 000 ")
    ax.set_title(title)
    fig.autofmt_xdate()

    fig.tight_layout()
    return fig, ax
