import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.dates import AutoDateLocator
from matplotlib.dates import DateFormatter
from sid.colors import get_colors

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_rki
from src.calculate_moments import smoothed_outcome_per_hundred_thousand_sim
from src.config import BLD


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def weekly_incidences_from_results(results, outcome):
    """Create the weekly incidences from a list of simulation runs.

    Args:
        results (list): list of DataFrames with the time series data from sid
            simulations.

    Returns:
        weekly_incidences (pandas.DataFrame): every column is the
            weekly incidence over time for one simulation run.
            The index are the dates of the simulation period.

    """
    weekly_incidences = []
    for res in results:
        weekly_incidences.append(
            smoothed_outcome_per_hundred_thousand_sim(
                df=res,
                outcome=outcome,
                take_logs=False,
                window=7,
                center=False,
            )
            * 7
        )
    weekly_incidences = pd.concat(weekly_incidences, axis=1)
    weekly_incidences.columns = range(len(results))
    return weekly_incidences


def plot_incidences(incidences, n_single_runs, title, name_to_label, rki=False):
    """Plot incidences.

    Args:
        incidences (dict): keys are names of the scenarios,
            values are dataframes where each column is the
            incidence of interest of one run
        n_single_runs (int): number of individual runs to
            visualize to show statistical uncertainty.
        title (str): plot title.
        rki (bool): Whether to plot the rki data.

    Returns:
        fig, ax

    """
    colors = get_colors("ordered", len(incidences))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in zip(incidences, colors):
        df = incidences[name]
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
                alpha=0.2,
            )
    if rki is not False:
        rki_data = pd.read_pickle(BLD / "data" / "processed_time_series" / "rki.pkl")
        rki_dates = rki_data.index.get_level_values("date")
        keep_dates = sorted(x for x in rki_dates if x in dates)
        cropped_rki = rki_data.loc[keep_dates]
        national_data = cropped_rki.groupby("date").sum()
        if rki == "new_known_case":
            rki_col = "newly_infected"
            label = "RKI Fallzahlen"
        elif rki == "newly_infected":
            rki_col = "upscaled_newly_infected"
            label = "DunkelzifferRadar Schätzung der \ntatsächlichen Inzidenz"
        else:
            raise ValueError(f"No matching RKI variable found to {rki}")

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

    fig, ax = style_plot(fig, ax)
    ax.set_ylabel("Geglättete wöchentliche \nNeuinfektionen pro 100 000")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(-0.0, -0.5, 1, 0.2), ncol=2)
    return fig, ax


def style_plot(fig, ax):
    ax.set_ylabel("")
    ax.set_xlabel("Datum")
    n_days = ax.get_xlim()[1] - ax.get_xlim()[0]
    date_form = DateFormatter("%d.%m") if n_days < 100 else DateFormatter("%m/%Y")
    ax.xaxis.set_major_formatter(date_form)
    loc = AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(loc)
    ax.grid(axis="y")
    sns.despine()
    return fig, ax
