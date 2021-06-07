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


@pytask.mark.depends_on(
    {
        "empirical": BLD / "data" / "empirical_data_for_plotting.pkl",
        # "plotting.py": SRC / "plotting" / "plotting.py",
    }
)
@pytask.mark.produces(
    {
        "fig": BLD / "figures" / "data" / "case_fatality_rate.pdf",
        "data": BLD / "tables" / "case_fatality_rate.csv",
        "mean_cfrs": BLD / "tables" / "mean_case_fatality_rates.csv",
    }
)
def task_create_and_plot_the_case_fatality_rate(depends_on, produces):
    empirical = pd.read_pickle(depends_on["empirical"])
    # forward date detected cases by the time it takes to die of Covid
    forwarded_cases = empirical["new_known_case"].tshift(28, freq="D")
    case_fatality_rate = empirical["newly_deceased"] / forwarded_cases
    case_fatality_rate = case_fatality_rate[PLOT_START_DATE:PLOT_END_DATE]
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    sns.lineplot(x=case_fatality_rate.index, y=case_fatality_rate)
    fig, ax = style_plot(fig, ax)
    fig.savefig(produces["fig"])
    case_fatality_rate.round(4).to_csv(produces["data"])

    mean_cfrs = pd.Series(
        {
            "oct_to_feb_cfr": case_fatality_rate["2020-10-01":"2021-01-31"].mean(),
            "march_to_june_cfr": case_fatality_rate["2021-03-01":"2021-05-31"].mean(),
        }
    )
    mean_cfrs.to_csv(produces["mean_cfrs"])
