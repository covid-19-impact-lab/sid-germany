import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
from sid.colors import get_colors

from src.calculate_moments import smoothed_outcome_per_hundred_thousand_sim


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
        results (list): list of DataFrames with the

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


def plot_incidences(incidences, n_single_runs, title):
    """Plot incidences.

    Args:
        incidences (dict): keys are names of the scenarios,
            values are dataframes where each column is the
            incidence of interest of one run
        n_single_runs (int): number of individual runs to
            visualize to show statistical uncertainty.
        title (str): plot title.

    Returns:
        fig, ax

    """
    colors = get_colors("ordered", len(incidences))
    fig, ax = plt.subplots(figsize=(6, 4))
    name_to_label = {
        "baseline": "Tatsächliche Home Office-Quote (14%)",
        "1_pct_more": "1 Prozent Mehr Home Office",
        "1st_lockdown_weak": "Home Office wie im Frühjahrslockdown, untere Grenze (25%)",  # noqa: E501
        "1st_lockdown_strict": "Home Office wie im Frühjahrslockdown, obere Grenze (35%)",  # noqa: E501
        "full_potential": "Volles Ausreizen des Home Office Potenzials (55%)",
        "november_baseline": "Home Office auf dem Niveau von November (15%)",
        "mobility_data_baseline": "Home Office auf dem Niveau von Anfang Januar (25%)",
    }
    for name, color in zip(incidences, colors):
        df = incidences[name]
        dates = df.index
        sns.lineplot(
            x=dates,
            y=df.mean(axis=1),
            ax=ax,
            color=color,
            label=name_to_label[name],
            linewidth=2.5,
            alpha=0.8,
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

    ax.set_ylabel("")
    ax.set_xlabel("Datum")
    date_form = DateFormatter("%d.%m")
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    ax.set_ylabel("Geglättete wöchentliche \nNeuinfektionen pro 100 000")
    ax.grid(axis="y")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(-0.0, -0.5, 1, 0.2))
    fig.tight_layout()

    return fig, ax
