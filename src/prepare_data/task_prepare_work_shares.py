import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


def clean_work_shares(path):
    work_shares = pd.read_csv(path)
    for col in ["men", "women"]:
        work_shares[col] = work_shares[col].astype(float) / 100
    work_shares[["age_lower", "age_upper"]] = work_shares["age_group"].apply(
        lambda x: interval_age_group(x)
    )
    work_shares.drop(columns=["age_group"], inplace=True)
    work_shares.rename(columns={"men": "male", "women": "female"}, inplace=True)

    work_shares["interval"] = work_shares.apply(
        lambda x: pd.Interval(x["age_lower"], x["age_upper"]), axis=1
    )
    work_shares.set_index("interval", inplace=True)

    return work_shares


def interval_age_group(x):
    if "-" in x:
        tup = x.split("-")
    elif ">=" in x:
        tup = x[2:], np.inf
    else:
        tup = float(x), float(x)
    tup = (float(tup[0]) - 0.5, float(tup[1]) + 0.5)
    return pd.Series(tup, index=["age_lower", "age_upper"])


@pytask.mark.depends_on(
    {
        "data": SRC
        / "original_data"
        / "population_structure"
        / "share_working_by_gender_2018.csv",
        "config": SRC / "config.py",
    }
)
@pytask.mark.produces(BLD / "data" / "population_structure" / "working_shares.pkl")
def task_prepare_work_shares(depends_on, produces):
    work_by_age_and_gender = clean_work_shares(depends_on["data"])
    work_by_age_and_gender.to_pickle(produces)
