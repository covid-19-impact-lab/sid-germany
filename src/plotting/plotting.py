from typing import Optional

import matplotlib.dates as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import sid

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.config import BLD
from src.config import SUMMER_SCENARIO_START
from src.simulation.scenario_config import get_named_scenarios


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def create_scenario_to_color():
    """Create the dictionary mapping scenario names to colors.

    Baseline scenarios get the typical sid blue. Vaccines scenarios get other blues.
    Rapid test scenarios get purple-red tones.
    School scenarios get the warm part of the ordered color map.

    """
    scenario_to_color = {
        "fall_baseline": "#547482",
        "summer_baseline": "#547482",
        # vaccines
        "spring_without_vaccines": "steelblue",
        "spring_with_more_vaccines": "dodgerblue",
        # rapid tests
        "spring_without_rapid_tests": "#5d4042",
        "spring_without_work_rapid_tests": "#a98286",
        "spring_without_school_rapid_tests": "#e0b1a3",
        "spring_with_mandatory_work_rapid_tests": "#b3563b",
        # school scenarios
        "spring_emergency_care_after_easter_without_school_rapid_tests": "#F1B05D",
        "spring_educ_open_after_easter_without_tests": "#3C2030",
        "spring_educ_open_after_easter_with_normal_tests": "#6c4a4d",
        "spring_open_educ_after_easter_with_tests_every_other_day": "#c87259",
        "spring_open_educ_after_easter_with_daily_tests": "#EE8445",
        # summer scenarios
        "summer_reduced_test_demand": "#3C2030",
        "summer_strict_home_office": "darkcyan",
        "summer_more_rapid_tests_at_work": "firebrick",
        "summer_optimistic_vaccinations": "dodgerblue",
        "summer_educ_open": "#6c4a4d",
    }

    all_scenarios = get_named_scenarios().keys()
    color_missing = [x for x in all_scenarios if x not in scenario_to_color.keys()]
    assert len(color_missing) == 0, "Some scenarios have no color: " + "\n\t".join(
        color_missing
    )
    return scenario_to_color


def calculate_virus_strain_shares(results):
    """Create the weekly incidences from a list of simulation runs.

    Args:
        results (list): list of DataFrames with the time series data from sid
            simulations.

    Returns:
        virus_strain_shares (pandas.DataFrame): every column is the
            weekly incidence over time for one simulation run.
            The index are the dates of the simulation period.

    """
    to_concat = []
    for res in results:
        new_known_case = res[res["new_known_case"]]
        n_infected_per_day = new_known_case["date"].value_counts().compute()
        grouped = new_known_case.groupby("date")
        # date and strain as MultiIndex
        n_strain_per_day = grouped["virus_strain"].value_counts()
        n_strain_per_day = n_strain_per_day.compute()
        n_strain_per_day = n_strain_per_day.unstack()
        share_strain_per_day = n_strain_per_day.divide(n_infected_per_day, axis=0)
        to_concat.append(share_strain_per_day.stack())
    strain_shares = pd.concat(to_concat, axis=1).unstack()
    strain_shares = strain_shares.swaplevel(axis=1)
    strain_shares.index.name = "date"
    return strain_shares


def plot_incidences(
    incidences,
    title,
    name_to_label,
    n_single_runs: Optional[int] = None,
    rki=False,
    plot_scenario_start=False,
    fig=None,
    ax=None,
):
    """Plot incidences.

    Args:
        incidences (dict): keys are names of the scenarios, values are dataframes where
            each column is the incidence of interest of one run
        title (str): plot title.
        n_single_runs (Optional[int]): Number of individual runs with different seeds
            visualize to show statistical uncertainty. Passing ``None`` will plot all
            runs.
        rki (bool): Whether to plot the rki data.
        plot_scenario_start (bool): whether to plot the scenario_start

    Returns:
        fig, ax

    """
    if fig is None and ax is None:
        if len(incidences) <= 4:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
    colors = create_scenario_to_color()
    for name, df in incidences.items():
        color = colors[name]
        dates = df.index
        sns.lineplot(
            x=dates,
            y=df.mean(axis=1),
            ax=ax,
            color=color,
            label=name_to_label[name] if name in name_to_label else name,
            linewidth=2.0,
            alpha=0.6,
        )
        # plot individual runs to visualize statistical uncertainty
        for run in df.columns[:n_single_runs]:
            sns.lineplot(
                x=dates,
                y=df[run],
                ax=ax,
                color=color,
                linewidth=0.5,
                alpha=0.1,
            )
    if rki:
        rki_data = pd.read_pickle(BLD / "data" / "processed_time_series" / "rki.pkl")
        rki_dates = rki_data.index.get_level_values("date")
        keep_dates = rki_dates.intersection(dates).unique().sort_values()
        cropped_rki = rki_data.loc[keep_dates]
        national_data = cropped_rki.groupby("date").sum()
        rki_col = "newly_infected"
        label = "official case numbers"

        weekly_smoothed = (
            smoothed_outcome_per_hundred_thousand_rki(
                df=national_data,
                outcome=rki_col,
                take_logs=False,
                window=7,
            )
            * 7
        )
        sns.lineplot(
            x=weekly_smoothed.index, y=weekly_smoothed, ax=ax, color="k", label=label
        )
    if plot_scenario_start:
        ax.axvline(
            pd.Timestamp(SUMMER_SCENARIO_START),
            label="scenario start",
            color="darkgrey",
        )

    fig, ax = style_plot(fig, ax)
    ax.set_ylabel("smoothed weekly incidence")
    ax.set_title(title)
    x, y, width, height = 0.0, -0.3, 1, 0.2
    ax.legend(loc="upper center", bbox_to_anchor=(x, y, width, height), ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_share_known_cases(share_known_cases, title):
    n_groups = share_known_cases.index.get_level_values("age_group_rki").nunique()
    colors = sid.get_colors("ordered", n_groups)
    sns.set_palette(colors)
    fig, ax = plt.subplots(figsize=(10, 5))

    for col in share_known_cases:
        alpha = 0.6 if col == "mean" else 0.2
        linewidth = 2.5 if col == "mean" else 1
        sns.lineplot(
            data=share_known_cases.reset_index(),
            x="date",
            y=col,
            hue="age_group_rki",
            linewidth=linewidth,
            alpha=alpha,
        )

    fig, ax = style_plot(fig, ax)
    ax.set_title(title)

    # Reduce the legend to have each age group only once and move it to below the plot
    x, y, width, height = 0.0, -0.3, 1, 0.2
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:n_groups],
        labels[:n_groups],
        loc="upper center",
        bbox_to_anchor=(x, y, width, height),
        ncol=n_groups,
    )

    fig.tight_layout()

    # undo side effect on color palette
    sns.color_palette()
    return fig, ax


def style_plot(fig, axes):
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax in axes:
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.grid(axis="y")
        ax = format_date_axis(ax)
    sns.despine()
    return fig, ax


def format_date_axis(ax):
    ax.xaxis.set_major_locator(dt.MonthLocator())
    # for month and year use "%b %Y"
    ax.xaxis.set_major_formatter(dt.DateFormatter("%B"))
    ax.xaxis.set_minor_locator(dt.DayLocator())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    return ax
