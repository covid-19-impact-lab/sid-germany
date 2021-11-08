import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.testing.shared import get_date_from_year_and_week


STRAIN_FILES = {
    "rki_strains": BLD / "data" / "virus_strains" / "rki_strains.csv",
    "virus_shares_dict": BLD / "data" / "virus_strains" / "virus_shares_dict.pkl",
}


@pytask.mark.depends_on(
    {
        "rki": SRC / "original_data" / "virus_strains_rki.csv",
        "testing_shared.py": SRC / "testing" / "shared.py",
    }
)
@pytask.mark.produces(STRAIN_FILES)
def task_prepare_virus_variant_data(depends_on, produces):
    rki = pd.read_csv(depends_on["rki"])
    rki = _prepare_rki_data(rki)
    rki.to_csv(produces["rki_strains"])

    b117 = rki["share_b117"]
    b117.name = "b117"

    delta = rki["share_delta"]
    delta.name = "delta"

    virus_shares = {
        "base_strain": 1 - b117 - delta,
        "b117": b117,
        "delta": delta,
    }
    pd.to_pickle(virus_shares, produces["virus_shares_dict"])


def _prepare_rki_data(df):
    df = df[df["week"].notnull()].copy(deep=True)
    df["year"] = 2021
    df["date"] = df.apply(get_date_from_year_and_week, axis=1)
    df = df.set_index("date").astype(float)
    for col in df:
        if col.startswith("pct_"):
            df[f"share_{col.replace('pct_', '')}"] = df[col] / 100
    share_cols = [col for col in df if col.startswith("share_")]
    df = df[share_cols]
    dates = pd.date_range(df.index.min(), df.index.max())
    # no division by 7 necessary because the data only contains shares.
    df = df.reindex(dates).interpolate()
    # add zero part for 2020
    df = df.reindex(pd.date_range("2020-03-01", df.index.max()), fill_value=0.0)
    df.index.name = "date"
    return df
