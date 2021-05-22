import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import SRC
from src.plotting.plotting import style_plot
from src.simulation.seasonality import create_seasonality_series

_MODULE_DEPENDENCIES = {
    "config": SRC / "config.py",
    "plotting": SRC / "plotting" / "plotting.py",
    "seasonality": SRC / "simulation" / "seasonality.py",
}

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


@pytask.mark.depends_on(_MODULE_DEPENDENCIES)
@pytask.mark.produces(BLD / "policies" / "seasonality.png")
def task_plot_seasonality(produces):
    dates = pd.date_range("2020-01-01", "2021-06-01")
    seasonality_series = create_seasonality_series(dates, 0.2)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=seasonality_series.index, y=seasonality_series, ax=ax)
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    fig.savefig(
        produces, dpi=200, transparent=False, facecolor="w", bbox_inches="tight"
    )
