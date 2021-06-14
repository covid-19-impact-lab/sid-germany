import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import PLOT_SIZE

_PRODUCT_PATH = (
    BLD
    / "figures"
    / "data"
    / "distributions_of_the_number_of_contacts"
    / "household.pdf"
)


@pytask.mark.depends_on(BLD / "data" / "initial_states.parquet")
@pytask.mark.produces(_PRODUCT_PATH)
def task_plot_n_hh_contacts(depends_on, produces):
    states = pd.read_parquet(depends_on)
    hh_sizes = states.groupby("hh_id").size()
    n_hh_contacts = (hh_sizes - 1).value_counts(normalize=True)
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.bar(x=n_hh_contacts.index, height=n_hh_contacts)
    sns.despine()
    fig.savefig(produces)
