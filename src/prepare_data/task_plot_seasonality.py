import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.config import PLOT_START_DATE
from src.config import SRC
from src.plotting.plotting import style_plot
from src.simulation.seasonality import create_seasonality_series

_DEPENDENCIES = {
    "plotting.py": SRC / "plotting" / "plotting.py",
    "seasonality.py": SRC / "simulation" / "seasonality.py",
    "params": BLD / "params.pkl",
}


@pytask.mark.depends_on(_DEPENDENCIES)
@pytask.mark.produces(BLD / "figures" / "data" / "seasonality.pdf")
def task_plot_seasonality(depends_on, produces):
    params = pd.read_pickle(depends_on["params"])
    tup = ("seasonality_effect", "seasonality_effect")
    weak_seasonality = params.loc[(*tup, "weak"), "value"]
    strong_seasonality = params.loc[(*tup, "strong"), "value"]

    dates = pd.date_range(PLOT_START_DATE, PLOT_END_DATE)
    weak_seasonality_series = create_seasonality_series(dates, weak_seasonality)
    strong_seasonality_series = create_seasonality_series(dates, strong_seasonality)

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.lineplot(
        x=weak_seasonality_series.index,
        y=weak_seasonality_series,
        ax=ax,
        label="seasonality of household, work and school contacts",
        linewidth=4,
        alpha=0.6,
    )
    sns.lineplot(
        x=strong_seasonality_series.index,
        y=strong_seasonality_series,
        ax=ax,
        label="seasonality of other (including leisure) contacts",
        linewidth=4,
        alpha=0.6,
    )

    fig, ax = style_plot(fig, ax)
    ax.set_ylabel("degree of infectiousness due to seasonality")
    fig.tight_layout()
    fig.savefig(produces, bbox_inches="tight")
    plt.close()
