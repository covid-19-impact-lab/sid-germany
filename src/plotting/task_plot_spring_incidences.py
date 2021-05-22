import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.config import SRC
from src.plotting.plotting import style_plot


@pytask.mark.depends_on(
    {
        "rki": BLD / "data" / "processed_time_series" / "rki.pkl",
        "config.py": SRC / "config.py",
        "plotting.py": SRC / "plotting" / "plotting.py",
    }
)
@pytask.mark.produces(BLD / "data" / "processed_time_series" / "spring_incidences.png")
def task_plot_spring_incidences(depends_on, produces):
    rki = pd.read_pickle(depends_on["rki"])
    rki_n_cases = rki.groupby("date")["newly_infected"].sum()
    rki_incidence = rki_n_cases * 100_000 / POPULATION_GERMANY
    smoothed_incidence = rki_incidence.rolling(7, center=True).sum()
    smoothed_incidence = smoothed_incidence.dropna()

    start_date = pd.Timestamp("2021-02-01")
    plot_data = smoothed_incidence.loc[start_date:].reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=plot_data, x="date", y="newly_infected")
    fig, ax = style_plot(fig, ax)
    ax.set_title("Weekly Incidence in Germany in Spring 2021")
    fig.savefig(produces, dpi=200, transparent=False, facecolor="w")
