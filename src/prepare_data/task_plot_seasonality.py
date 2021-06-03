import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import SRC
from src.plotting.plotting import style_plot
from src.simulation.seasonality import create_seasonality_series

_DEPENDENCIES = {
    "plotting.py": SRC / "plotting" / "plotting.py",
    "seasonality.py": SRC / "simulation" / "seasonality.py",
    "params": BLD / "params.pkl",
}

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.produces(BLD / "figures" / "data" / "seasonality.png")
def task_plot_seasonality(depends_on, produces):
    params = pd.read_pickle(depends_on["params"])
    tup = ("seasonality_effect", "seasonality_effect")
    weak_seasonality = params.loc[(*tup, "weak"), "value"]
    strong_seasonality = params.loc[(*tup, "strong"), "value"]

    dates = pd.date_range("2020-09-01", "2021-06-01")
    weak_seasonality_series = create_seasonality_series(dates, weak_seasonality)
    strong_seasonality_series = create_seasonality_series(dates, strong_seasonality)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        x=weak_seasonality_series.index,
        y=weak_seasonality_series,
        ax=ax,
        label="seasonality of household, work and school contacts",
    )
    sns.lineplot(
        x=strong_seasonality_series.index,
        y=strong_seasonality_series,
        ax=ax,
        label="seasonality of other (including leisure) contacts",
    )

    fig, ax = style_plot(fig, ax)
    ax.set_title("Strong and Weak Seasonality Effects Over Time")
    fig.tight_layout()
    fig.savefig(
        produces, dpi=200, transparent=False, facecolor="w", bbox_inches="tight"
    )
