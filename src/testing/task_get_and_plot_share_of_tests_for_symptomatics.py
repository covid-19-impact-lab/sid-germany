import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.simulation.plotting import style_plot
from src.testing.shared import convert_weekly_to_daily
from src.testing.shared import (
    get_date_from_year_and_week,
)

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.figsize": (12, 3.5),
    }
)

OUT_PATH = BLD / "data" / "testing"


@pytask.mark.depends_on(BLD / "data" / "raw_time_series" / "test_distribution.xlsx")
@pytask.mark.produces(
    {
        "data": OUT_PATH / "characteristics_of_the_tested.csv",
        "mean_age": OUT_PATH / "mean_age_of_tested.png",
        "share_with_symptom_status": OUT_PATH
        / "share_of_tested_with_symptom_status.png",
        "symptom_shares": OUT_PATH / "share_of_symptomatic_among_tests.png",
    }
)
def task_prepare_characteristics_of_the_tested(depends_on, produces):
    df = pd.read_excel(depends_on, sheet_name="Klinische_Aspekte", header=1)

    df = _clean_data(df)
    df = convert_weekly_to_daily(df.reset_index(), divide_by_7_cols=[])
    df.to_csv(produces["data"])

    fig, ax = _plot_df_column(df, "mean_age")
    fig.tight_layout()
    fig.savefig(produces["mean_age"])
    fig, ax = _plot_df_column(df, "share_with_symptom_status")
    fig.tight_layout()
    fig.savefig(produces["share_with_symptom_status"])
    symptom_shares = [
        "share_symptomatic_lower_bound",
        "share_symptomatic_among_known",
        "share_symptomatic_upper_bound",
    ]
    fig, ax = _plot_df_column(df, symptom_shares, "Symptomatic Shares")
    fig.tight_layout()
    fig.savefig(produces["symptom_shares"])


def _clean_data(df):
    share_sym_de = "Anteil keine, bzw. keine für COVID-19 bedeutsamen Symptome"
    column_translation = {
        "Meldejahr": "year",
        "MW": "week",
        "Fälle gesamt": "n_total_cases",
        "Mittelwert Alter (Jahre)": "mean_age",
        "Anzahl mit Angaben zu Symptomen": "n_with_symptom_status",
        share_sym_de: "share_asymptomatic_among_known",
    }

    df = df.rename(columns=column_translation)
    df = df[column_translation.values()]
    df["date"] = df.apply(get_date_from_year_and_week, axis=1)
    df = df.set_index("date")
    df["share_with_symptom_status"] = df["n_with_symptom_status"] / df["n_total_cases"]
    df["share_symptomatic_among_known"] = 1 - df["share_asymptomatic_among_known"]
    keep = [
        "mean_age",
        "share_with_symptom_status",
        "share_asymptomatic_among_known",
        "share_symptomatic_among_known",
    ]
    df = df[keep]

    df["share_without_symptom_status"] = 1 - df["share_with_symptom_status"]
    # The lower bound on the share of symptomatics is assuming everyone without
    # symptom status was asymptomatic
    df["share_symptomatic_lower_bound"] = (
        df["share_symptomatic_among_known"] * df["share_with_symptom_status"]
    )
    df["share_symptomatic_upper_bound"] = (
        df["share_symptomatic_lower_bound"] + df["share_without_symptom_status"]
    )
    return df


def _plot_df_column(df, cols, title=None):
    if isinstance(cols, str):
        cols = [cols]
    fig, ax = plt.subplots()
    for col in cols:
        label = col.replace("_", " ").title()
        sns.lineplot(x=df["date"], y=df[col], ax=ax, label=label)
    if title is not None:
        ax.set_title(title)
    elif len(cols) == 1:
        ax.set_title(label)
    style_plot(fig, ax)
    return fig, ax