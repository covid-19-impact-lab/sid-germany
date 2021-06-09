import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_END_DATE
from src.config import PLOT_SIZE
from src.config import PLOT_START_DATE
from src.plotting.plotting import style_plot


@pytask.mark.depends_on(BLD / "data" / "processed_time_series" / "r_effective.pkl")
@pytask.mark.produces(BLD / "figures" / "data" / "r_effective_acc_to_rki.pdf")
def task_plot_r_effective_acc_to_rki(depends_on, produces):
    df = pd.read_pickle(depends_on)
    to_plot = df.loc[PLOT_START_DATE:PLOT_END_DATE, "Schätzer_7_Tage_R_Wert"]
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.lineplot(x=to_plot.index, y=to_plot)
    fig, ax = style_plot(fig, ax)
    fig.savefig(produces)
