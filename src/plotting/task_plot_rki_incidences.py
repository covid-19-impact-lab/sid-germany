import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.config import PLOT_START_DATE
from src.config import POPULATION_GERMANY
from src.config import SRC
from src.plotting.plotting import style_plot


@pytask.mark.depends_on(
    {
        "rki": BLD / "data" / "processed_time_series" / "rki.pkl",
        "plotting.py": SRC / "plotting" / "plotting.py",
    }
)
@pytask.mark.produces(
    {
        "spring": BLD / "figures" / "data" / "official_case_numbers_in_spring.pdf",
        "overall": BLD / "figures" / "data" / "official_case_numbers.pdf",
    }
)
def task_plot_spring_incidences(depends_on, produces):
    rki = pd.read_pickle(depends_on["rki"])
    rki_n_cases = rki.groupby("date")["newly_infected"].sum()
    rki_incidence = rki_n_cases * 100_000 / POPULATION_GERMANY
    smoothed_incidence = rki_incidence.rolling(7, center=True).sum()
    smoothed_incidence = smoothed_incidence.dropna()

    plot_data = smoothed_incidence.loc[PLOT_START_DATE:PLOT_END_DATE]

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.lineplot(x=plot_data.index, y=plot_data, linewidth=2.5)
    fig, ax = style_plot(fig, ax)
    fig.savefig(produces["overall"])
    plt.close()

    spring_start = pd.Timestamp("2021-02-01")
    spring_data = plot_data.loc[spring_start:]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=spring_data.index, y=spring_data, linewidth=2.5)
    fig, ax = style_plot(fig, ax)
    fig.savefig(produces["spring"])
    plt.close()
