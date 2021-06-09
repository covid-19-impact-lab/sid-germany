import numpy as np
import pandas as pd
import pytask

from src.config import BLD


@pytask.mark.depends_on(BLD / "data" / "raw_time_series" / "reproduction_number.csv")
@pytask.mark.produces(BLD / "data" / "processed_time_series" / "r_effective.pkl")
def task_prepare_rki_r_effective_data(depends_on, produces):
    df = pd.read_csv(depends_on, delimiter=";")
    df = df[df["Datum"].isnull().cumsum() == 0]
    df["date"] = pd.to_datetime(df["Datum"], dayfirst=True)
    df = df.replace(".", np.nan)
    to_float = [
        "Schätzer_Reproduktionszahl_R",
        "UG_PI_Reproduktionszahl_R",
        "OG_PI_Reproduktionszahl_R",
        "Schätzer_7_Tage_R_Wert",
        "UG_PI_7_Tage_R_Wert",
        "OG_PI_7_Tage_R_Wert",
    ]
    for col in to_float:
        df[col] = df[col].str.replace(",", ".").astype(float)
    df = df.set_index("date").sort_index()
    df.to_pickle(produces)
