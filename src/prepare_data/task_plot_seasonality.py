import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.simulation.seasonality import seasonality_model
import pytask
from src.config import BLD


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


@pytask.mark.produces(BLD / "policies" / "seasonality.png")
def task_plot_seasonality(produces):
    params = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [("seasonality_effect", "seasonality_effect", "seasonality_effect")]
        ),
        columns=["value"],
        data=[0.2],
    )
    dates = pd.date_range("2020-01-01", "2021-06-01")
    seasonality_series = seasonality_model(params, dates, 4949)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=seasonality_series.index, y=seasonality_series, ax=ax)
    fig.tight_layout()
    fig.savefig(
        produces, dpi=200, transparent=False, facecolor="w", bbox_inches="tight"
    )
