import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(
    {
        "data": SRC
        / "original_data"
        / "population_structure"
        / "eu_age_structure.tsv.gz",
        "config.py": SRC / "config.py",
    }
)
@pytask.mark.produces(BLD / "data" / "population_structure" / "eu_age_structure.pkl")
def task_eu_age_distribution(depends_on, produces):
    age_data_path = Path(depends_on["data"])
    with gzip.open(age_data_path, "rb") as f:
        data = pd.read_csv(f, sep=",|\t", engine="python")

    data = data[data["sex"] == "T"][["age", r"geo\time", "2019 "]]
    countries = ["BE", "DE_TOT", "FI", "IT", "LU", "NL", "PL", "UK"]
    data = data[data[r"geo\time"].isin(countries)]
    data = data[~data["age"].isin(["UNK"])]

    total_pop = data[data["age"] == "TOTAL"].set_index(r"geo\time")["2019 "].astype(int)
    # Y_OPEN is people above 99 years old which was not asked in the sample data and
    # who are very few.
    age_data = data[~data["age"].isin(["TOTAL", "Y_OPEN"])].copy()
    age_data["age"].replace({"Y_LT1": "Y0"}, inplace=True)
    age_data["age"] = age_data["age"].str[1:].astype(int)
    age_data = age_data.set_index([r"geo\time", "age"])
    age_data = age_data.unstack()["2019 "].astype(int).T
    age_data = age_data / total_pop
    age_data.columns.name = "country"
    age_data.to_pickle(produces)


@pytask.mark.depends_on(
    SRC / "original_data" / "population_structure" / "eu_hh_sizes.zip"
)
@pytask.mark.produces(BLD / "data" / "population_structure" / "eu_hh_size_shares.pkl")
def task_eu_hh_size_distribution(depends_on, produces):
    hh_size_path = Path(depends_on)
    with gzip.open(hh_size_path, "rb") as f:
        data = pd.read_csv(f, sep=",|\t", engine="python")

    data["hh_size"] = data["n_person"].replace({"GE6": pd.Interval(5, np.inf)})

    # we only have West Germany. Use it for Germany as a whole.
    data["country"] = data[r"geo\time"].replace("DE", "DE_TOT")

    countries = ["BE", "DE_TOT", "FI", "IT", "LU", "NL", "PL", "UK"]
    data = data.set_index(["country", "hh_size"])["2018 "]
    data = data.loc[countries].sort_index().astype(float)

    hh_size_shares = data.unstack().T / 100
    hh_size_shares.index = [
        int(x) if isinstance(x, str) else x for x in hh_size_shares.index
    ]

    hh_size_shares.to_pickle(produces)
