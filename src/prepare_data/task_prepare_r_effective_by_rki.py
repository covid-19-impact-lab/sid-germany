import numpy as np
import pandas as pd
import pytask

from src.config import BLD


@pytask.mark.depends_on(BLD / "data" / "raw_time_series" / "reproduction_number.xlsx")
@pytask.mark.produces(BLD / "data" / "processed_time_series" / "r_effective.pkl")
def task_prepare_rki_r_effective_data(depends_on, produces):
    df = pd.read_excel(depends_on, sheet_name="Nowcast_R")

    df["date"] = pd.to_datetime(df["Datum des Erkrankungsbeginns"], dayfirst=True)
    df = df.replace(".", np.nan)
    df = df.set_index("date").sort_index()

    try:
        r_effective = df["Punktschätzer des 7-Tage-R Wertes"]
    except KeyError:
        r_effective = df["Punktschätzer des 7-Tage-R-Wertes"]
    r_effective = r_effective.str.replace(",", ".").astype(float)
    r_effective.to_pickle(produces)
