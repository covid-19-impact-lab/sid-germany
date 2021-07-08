import pandas as pd
import pytask

from src.config import BLD


@pytask.mark.depends_on(BLD / "data" / "raw_time_series" / "reproduction_number.csv")
@pytask.mark.produces(BLD / "data" / "processed_time_series" / "r_effective.pkl")
def task_prepare_rki_r_effective_data(depends_on, produces):
    df = pd.read_csv(depends_on)

    df["date"] = pd.to_datetime(df["Datum"], yearfirst=True)
    df = df.set_index("date").sort_index()

    r_effective = df["PS_7_Tage_R_Wert"]
    r_effective.to_pickle(produces)
