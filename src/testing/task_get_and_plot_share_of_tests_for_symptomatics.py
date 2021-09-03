import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from sid import get_colors

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.config import PLOT_START_DATE
from src.config import SRC
from src.plotting.plotting import BLUE
from src.plotting.plotting import style_plot
from src.testing.shared import convert_weekly_to_daily
from src.testing.shared import get_date_from_year_and_week

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.figsize": (12, 3.5),
    }
)


@pytask.mark.depends_on(
    {
        "data": BLD / "data" / "raw_time_series" / "test_distribution.xlsx",
        "testing_shared.py": SRC / "testing" / "shared.py",
    }
)
@pytask.mark.produces(
    {
        "data": BLD / "data" / "testing" / "characteristics_of_the_tested.csv",
        "share_of_tests_for_symptomatics_series": BLD
        / "data"
        / "testing"
        / "share_of_tests_for_symptomatics_series.pkl",
        "mean_age": BLD / "data" / "testing" / "mean_age_of_tested.pdf",
        "share_with_symptom_status": BLD
        / "data"
        / "testing"
        / "share_of_tested_with_symptom_status.pdf",
        "symptom_shares": BLD
        / "figures"
        / "data"
        / "testing"
        / "share_of_pcr_tests_going_to_symptomatics.pdf",
        "used_share_pcr_going_to_symptomatic": BLD
        / "figures"
        / "data"
        / "testing"
        / "used_share_of_pcr_tests_going_to_symptomatics.pdf",
    }
)
def task_prepare_characteristics_of_the_tested(depends_on, produces):
    df = pd.read_excel(depends_on["data"], sheet_name="Klinische_Aspekte", header=2)

    df = _clean_data(df)
    df = convert_weekly_to_daily(df.reset_index(), divide_by_7_cols=[])

    plot_data = df[df["date"].between(PLOT_START_DATE, PLOT_END_DATE)]

    fig, ax = _plot_df_column(plot_data, "mean_age")
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    fig.savefig(produces["mean_age"])
    plt.close()

    fig, ax = _plot_df_column(plot_data, "share_with_symptom_status")
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    fig.savefig(produces["share_with_symptom_status"])
    plt.close()

    symptom_shares = [
        "share_symptomatic_lower_bound",
        "share_symptomatic_among_known",
        "share_symptomatic_upper_bound",
    ]

    df = df.set_index("date")
    to_concat = [df]
    for share in symptom_shares:
        extrapolated = _extrapolate_series_after_february(df[share])
        to_concat.append(extrapolated)
    df = pd.concat(to_concat, axis=1)

    colors = get_colors("categorical", len(symptom_shares))
    fig, ax = plt.subplots(figsize=PLOT_SIZE)

    for share, color in zip(symptom_shares, colors):
        extrapolated = f"{share}_extrapolated"
        sns.lineplot(x=df.index, y=df[share], ax=ax, color=color, label=share)
        sns.lineplot(x=df.index, y=df[extrapolated], ax=ax, color=color)
    fig.tight_layout()
    fig, ax = style_plot(fig, ax)
    fig.savefig(produces["symptom_shares"])
    plt.close()

    share_of_tests_for_symptomatics_series = df[
        [
            "share_symptomatic_lower_bound_extrapolated",
            "share_symptomatic_among_known_extrapolated",
        ]
    ].mean(axis=1)
    share_of_tests_for_symptomatics_series.to_pickle(
        produces["share_of_tests_for_symptomatics_series"]
    )

    df = df.reset_index().rename(columns={"index": "date"})
    df.to_csv(produces["data"])

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.lineplot(
        x=share_of_tests_for_symptomatics_series.index,
        y=share_of_tests_for_symptomatics_series,
        color=BLUE,
        linewidth=3.0,
        alpha=0.6,
    )
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    fig.savefig(produces["used_share_pcr_going_to_symptomatic"])


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


def _extrapolate_series_after_february(sr, end_date="2021-08-30"):
    end_date = pd.Timestamp(end_date)
    last_empirical_date = min(pd.Timestamp("2021-02-28"), sr.index.max())
    empirical_part = sr[:last_empirical_date]
    extension_index = pd.date_range(
        last_empirical_date + pd.Timedelta(days=1), end_date
    )
    extension_value = sr[
        last_empirical_date - pd.Timedelta(days=30) : last_empirical_date
    ].mean()
    extension = pd.Series(extension_value, index=extension_index)
    out = pd.concat([empirical_part, extension])
    out.name = f"{sr.name}_extrapolated"
    return out


def _plot_df_column(df, cols, title=None):
    if isinstance(cols, str):
        cols = [cols]
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    for col in cols:
        label = col.replace("_", " ").title()
        sns.lineplot(x=df["date"], y=df[col], ax=ax, label=label)
    if title is not None:
        ax.set_title(title)
    elif len(cols) == 1:
        ax.set_title(label)
    style_plot(fig, ax)
    return fig, ax
