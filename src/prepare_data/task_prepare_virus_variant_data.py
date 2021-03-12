import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns
from sid import get_colors

from src.config import BLD
from src.config import SRC
from src.simulation.plotting import style_plot
from src.testing.shared import get_date_from_year_and_week

OUT_PATH = BLD / "data" / "virus_strains"


@pytask.mark.depends_on(
    {
        "rki": SRC / "original_data" / "virus_strains_rki.csv",
        "cologne": SRC / "original_data" / "virus_strains_cologne.csv",
    }
)
@pytask.mark.produces(
    {
        "rki_strains": OUT_PATH / "rki_strains.csv",
        "co_daily": OUT_PATH / "cologne_strains_daily.csv",
        "co_weekly": OUT_PATH / "cologne_strains_weekly.csv",
        "b117": OUT_PATH / "b117.pkl",
        "b1351": OUT_PATH / "b1351.pkl",
        "fig": OUT_PATH / "figures" / "averaged_strain_shares.png",
    }
)
def task_prepare_virus_variant_data(depends_on, produces):
    fig_path = produces["fig"].parent
    rki = pd.read_csv(depends_on["rki"])
    rki = _prepare_rki_data(rki)
    rki.to_csv(produces["rki_strains"])

    co_daily = pd.read_csv(depends_on["cologne"])
    co_daily = _prepare_co_data(co_daily)
    co_daily.to_csv(produces["co_daily"])

    co_weekly = _make_cologne_data_weekly(co_daily)
    co_weekly.to_csv(produces["co_weekly"])

    for col in co_daily:
        fig, ax = _plot_cologne_data(co_daily=co_daily, co_weekly=co_weekly, col=col)
        path = fig_path / f"{col}_cologne.png"
        fig.savefig(path, dpi=200, transparent=False, facecolor="w")

    for col in co_weekly:
        fig, ax = _rki_vs_cologne_data(rki=rki, co_weekly=co_weekly, col=col)
        path = fig_path / f"{col}_rki_vs_cologne.png"
        fig.savefig(path, dpi=200, transparent=False, facecolor="w")

    strain_data = _merge_rki_and_cologne_data(rki, co_weekly)

    # average over rki and cologne shares
    b117 = strain_data.groupby("date")["share_b117"].mean()
    b117.to_pickle(produces["b117"])
    b1351 = strain_data.groupby("date")["share_b1351"].mean()
    b1351.to_pickle(produces["b1351"])

    fig, ax = _plot_final_shares(b117, b1351)
    fig.savefig(produces["fig"])


def _prepare_rki_data(rki):
    rki = rki[rki["week"].notnull()]
    rki["year"] = 2021
    rki["date"] = rki.apply(get_date_from_year_and_week, axis=1)
    rki = rki.set_index("date")
    as_float_cols = ["pct_b117", "pct_b1351", "n_tested_for_variants"]
    rki[as_float_cols] = rki[as_float_cols].astype(float)
    rki["share_b117"] = rki["pct_b117"] / 100
    rki["share_b1351"] = rki["pct_b1351"] / 100
    rki = rki[["share_b117", "share_b1351", "n_tested_for_variants"]]
    return rki


def _prepare_co_data(co):
    co = pd.read_csv(SRC / "original_data" / "virus_strains_cologne.csv")
    co = co[co["n_b117_cum"].notnull() & co["n_tests_positive_cum"].notnull()]
    co["date"] = pd.to_datetime(co["date"], dayfirst=True)
    keep_cols = ["n_b117_cum", "n_b1351_cum", "n_tests_positive_cum"]
    co = co.set_index("date")[keep_cols].astype(int)
    # As the latest date is always added on top, this checks that there are no typos.
    assert (
        co.index.is_monotonic_decreasing
    ), "Dates of the Cologne virus strain data are not monotonic. Typo?"
    co = co.sort_index()
    for col in co:
        assert (co[col].diff().dropna() >= 0).all(), col
        co[col.replace("_cum", "")] = co[col].diff()
    co = co.loc["2021-02-04":]
    co["share_b117"] = co["n_b117"] / co["n_tests_positive"]
    co["share_b1351"] = co["n_b1351"] / co["n_tests_positive"]
    co = co.rename(columns={"n_tests_positive": "n_tested_for_variants"})
    return co


def _make_cologne_data_weekly(co):
    co_small = co[["n_b117", "n_b1351", "n_tested_for_variants"]].reset_index()
    co_weekly = co_small.groupby(pd.Grouper(key="date", freq="W")).sum()
    # remove latest, incomplete week
    co_weekly = co_weekly[:-1]
    co_weekly["share_b117"] = co_weekly["n_b117"] / co_weekly["n_tested_for_variants"]
    co_weekly["share_b1351"] = co_weekly["n_b1351"] / co_weekly["n_tested_for_variants"]
    co_weekly = co_weekly[["n_tested_for_variants", "share_b117", "share_b1351"]]
    return co_weekly


def _rki_vs_cologne_data(rki, co_weekly, col):
    colors = get_colors("categorical", 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=rki.index, y=rki[col], color=colors[0], linewidth=2, label="RKI")
    sns.lineplot(
        x=co_weekly.index,
        y=co_weekly[col],
        color=colors[1],
        linewidth=2,
        label="Cologne",
    )
    nice_name = col.replace("_", " ").title()
    ax.set_title(f"{nice_name} Acc. to RKI and in Cologne")
    fig, ax = style_plot(fig, ax)
    return fig, ax


def _plot_cologne_data(co_daily, co_weekly, col):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = get_colors("categorical", 3)
    if col.endswith("_cum"):
        color = colors[0]
    elif col.startswith("share"):
        color = colors[1]
    else:
        color = colors[2]
    if col in co_weekly:
        sns.lineplot(
            x=co_weekly.index,
            y=co_weekly[col],
            color=color,
            linewidth=2,
            linestyle="--",
            label="weekly",
        )
        label = "daily"
    else:
        label = None
    sns.lineplot(
        x=co_daily.index, y=co_daily[col], color=color, linewidth=2, label=label
    )
    nice_name = col.replace("_", " ").title()
    ax.set_title(f"{nice_name} in Cologne")
    fig, ax = style_plot(fig, ax)
    return fig, ax


def _merge_rki_and_cologne_data(rki, co_weekly):
    rki = rki.copy(deep=True)
    co_weekly = co_weekly.copy(deep=True)

    co_weekly["source"] = "cologne"
    rki["source"] = "rki"
    co_weekly = co_weekly.set_index("source", append=True)
    rki = rki.set_index("source", append=True)

    strain_data = pd.concat([rki, co_weekly]).sort_index()
    return strain_data


def _plot_final_shares(b117, b1351):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = get_colors("categorical", 2)
    sns.lineplot(x=b117.index, y=b117, color=colors[0], linewidth=2, label="b117")
    sns.lineplot(x=b1351.index, y=b1351, color=colors[1], linewidth=2, label="b1351")
    ax.set_title("Share of Virus Variants Over Time")
    fig, ax = style_plot(fig, ax)
    return fig, ax
