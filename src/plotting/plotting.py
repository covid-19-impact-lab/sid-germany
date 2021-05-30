from typing import Optional

import matplotlib.dates as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.config import BLD
from src.config import SRC


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


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
    colors,
    n_single_runs: Optional[int] = None,
    empirical=False,
    scenario_starts=None,
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
        empirical (bool): Whether to plot the empirical data.
        scenario_start (list, optional): the scenario start points

    Returns:
        fig, ax

    """
    if fig is None and ax is None:
        if len(incidences) <= 4:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig, ax = plt.subplots(figsize=(6, 6))

    dates = list(incidences.values())[0].index
    if empirical in ["new_known_case", "newly_deceased"]:
        rki_data = pd.read_pickle(BLD / "data" / "processed_time_series" / "rki.pkl")
        rki_dates = rki_data.index.get_level_values("date")
        dates = rki_dates.intersection(dates).unique().sort_values()

    if colors is None:
        colors = [
            "#4e79a7",
            "#f28e2b",
            "#e15759",
            "#76b7b2",
            "#59a14f",
            "#edc948",
            "#b07aa1",
            "#9c755f",
        ]
    for (name, df), color in zip(incidences.items(), colors):
        sns.lineplot(
            x=dates,
            y=df.loc[dates].mean(axis=1),
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
                y=df.loc[dates, run],
                ax=ax,
                color=color,
                linewidth=0.5,
                alpha=0.1,
            )
    if empirical is not False:
        if empirical in ["new_known_case", "newly_deceased"]:
            rki_data = rki_data.groupby("date").sum()
            if empirical == "new_known_case":
                rki_col = "newly_infected"
            else:
                rki_col = empirical
            label = "official case numbers"

            weekly_smoothed = (
                smoothed_outcome_per_hundred_thousand_rki(
                    df=rki_data,
                    outcome=rki_col,
                    take_logs=False,
                    window=7,
                )
                * 7
            )
            sns.lineplot(
                x=dates, y=weekly_smoothed[dates], ax=ax, color="k", label=label
            )
        elif empirical == "ever_had_a_rapid_test":
            cosmo_share = pd.read_csv(
                SRC
                / "original_data"
                / "testing"
                / "cosmo_share_ever_had_a_rapid_test.csv",
                parse_dates=["date"],
                index_col="date",
            )
            label = (
                "share of Germans reporting to have ever done\n"
                "a rapid test acc. to the COSMO data"
            )
            sns.lineplot(
                x=cosmo_share.index,
                y=cosmo_share["share_ever_had_a_rapid_test"],
                label=label,
                ax=ax,
                color="k",
            )
        elif empirical == "last_rapid_test_in_the_last_week":
            cosmo_share = pd.read_csv(
                SRC
                / "original_data"
                / "testing"
                / "cosmo_selftest_frequency_last_four_weeks.csv",
                parse_dates=["date"],
                index_col="date",
            )
            weekly_or_more_cols = [
                "share_more_than_5_tests_per_week",
                "share_5_tests_per_week",
                "share_2-4_tests_per_week",
                "share_weekly",
            ]
            cosmo_share = cosmo_share[weekly_or_more_cols].sum(axis=1)
            label = (
                "share of Germans reporting to have done\n"
                "at least one self-administered rapid test per week\n"
                "within the last four weeks acc. to the COSMO data"
            )
            sns.lineplot(
                x=cosmo_share.index,
                y=cosmo_share["share_ever_had_a_rapid_test"],
                label=label,
                ax=ax,
                color="k",
            )
        else:
            raise ValueError(f"No known empirical equivalent for {empirical}.")

    if scenario_starts is not None:
        if isinstance(scenario_starts, (str, pd.Timestamp)):
            scenario_starts = [(scenario_starts, "scenario start")]

        for date, label in scenario_starts:
            ax.axvline(
                pd.Timestamp(date),
                label=label,
                color="darkgrey",
            )

    fig, ax = style_plot(fig, ax)
    ax.set_ylabel("smoothed weekly incidence")
    ax.set_title(title)
    x, y, width, height = 0.0, -0.3, 1, 0.2
    ax.legend(loc="upper center", bbox_to_anchor=(x, y, width, height), ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_share_known_cases(share_known_cases, title, plot_single_runs=False):
    colors = [
        "#4e79a7",  # blue
        "#76b7b2",  # light blue
        "#edc948",  # yellow
        "#f28e2b",  # orange
        "#e15759",  # red
        "#b07aa1",  # purple
    ]
    sns.set_palette(colors)
    fig, ax = plt.subplots(figsize=(10, 5))

    if plot_single_runs:
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

        handles, labels = ax.get_legend_handles_labels()
        # Reduce legend to have each age group only once and move it to below the plot
        x, y, width, height = 0.0, -0.3, 1, 0.2
        n_groups = share_known_cases.index.get_level_values("age_group_rki").nunique()
        ax.legend(
            handles[:n_groups],
            labels[:n_groups],
            loc="upper center",
            bbox_to_anchor=(x, y, width, height),
            ncol=n_groups,
        )

    else:
        sns.lineplot(
            data=share_known_cases.reset_index(),
            x="date",
            y="mean",
            hue="age_group_rki",
            linewidth=2.5,
            alpha=0.6,
        )

    fig, ax = style_plot(fig, ax)
    ax.set_title(title)

    fig.tight_layout()

    # undo side effect on color palette
    sns.color_palette()
    return fig, ax


def plot_group_time_series(df, title, rki=None):
    """Plot a time series by group with more than one run.

    Args:
        df (pandas.DataFrame): index levels are dates and group identifiers.
            There is one column for each simulation run.
        title (str): the title of the plot
        rki (pandas.Series, optional): Series with the RKI data. Must have the same
            index as df.

    """
    df = df.swaplevel().copy(deep=True)
    groups = df.index.levels[0].unique()
    dates = df.index.levels[1].unique()

    n_rows = int(np.ceil(len(groups) / 2))
    fig, axes = plt.subplots(figsize=(12, n_rows * 3), nrows=n_rows, ncols=2)
    axes = axes.flatten()

    if "0-4" in groups:
        colors = [
            "#C89D64",
            "#F1B05D",
            "#EE8445",
            "#c87259",
            "#6c4a4d",
            "#3C2030",
        ]
    else:
        colors = ["#C89D64"] * len(groups)

    for group, ax in zip(groups, axes):
        plot_incidences(
            incidences={group: df.loc[group]},
            title=title.format(group=group),
            name_to_label={group: "simulated"},
            empirical=False,
            colors=colors,
            scenario_starts=None,
            fig=fig,
            ax=ax,
        )

        if rki is not None:
            rki_data = rki.loc[dates, group].reset_index()
            sns.lineplot(
                x=rki_data["date"],
                y=rki_data[0],
                ax=ax,
                color="k",
                label="official case numbers",
            )

    return fig, axes


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


def shorten_dfs(dfs, empirical, plot_start=None):
    """Shorten all incidence DataFrames.

    All DataFrames are shortened to the shortest. In addition, if plot_start is given
    all DataFrames start at or after plot_start.

    Args:
        dfs (dict): keys are the names of the scenarios, values are the incidence
            DataFrames.
        empirical (str or bool): if not False and empirical is the name of an
            outcome that is in the RKI data, reduce the DataFrames to the time
            for which RKI data is available.
        plot_start (pd.Timestamp or None): earliest allowed start date for the plot

    Returns:
        shortened (dict): keys are the names of the scenarios, values are the shortened
            DataFrames.

    """
    shortened = {}

    start_date = max(df.index.min() for df in dfs.values())
    if plot_start is not None:
        start_date = max(plot_start, start_date)
    end_date = min(df.index.max() for df in dfs.values())
    if empirical and empirical in ["new_known_case", "newly_deceased"]:
        rki_data = pd.read_pickle(BLD / "data" / "processed_time_series" / "rki.pkl")
        end_date = min(end_date, rki_data.index.get_level_values("date").max())

    for name, df in dfs.items():
        shortened[name] = df.loc[start_date:end_date].copy(deep=True)

    return shortened


def create_nice_labels(names):
    name_to_label = {}
    for name in names:
        name_to_label[name] = make_name_nice(name)
    return name_to_label


def make_name_nice(name):
    """Make a scenario name nice.

    Args:
        name (str): name of the scenario

    Returns:
        nice_name (str): nice name of the scenario

    """
    replacements = [
        ("_", " "),
        (" with", "\n with"),
        ("fall", ""),
        ("spring", ""),
        ("summer", ""),
    ]
    nice_name = name
    for old, new in replacements:
        nice_name = nice_name.replace(old, new)
    nice_name = nice_name.lstrip("\n")
    return nice_name
