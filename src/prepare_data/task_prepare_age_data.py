import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.shared import create_age_groups_rki


@pytask.mark.depends_on(
    {
        "data": SRC / "original_data" / "population_structure" / "altersjahre.csv",
        "shared.py": SRC / "shared.py",
    }
)
@pytask.mark.produces(
    [
        BLD / "data" / "population_structure" / "age_groups.parquet",
        BLD / "data" / "population_structure" / "age_groups_rki.pkl",
    ]
)
def task_prepare_age_data_de(depends_on, produces):
    raw = (
        pd.read_csv(
            depends_on["data"], sep=";", skiprows=6, skipfooter=5, engine="python"
        )["31.12.2018"]
        .reset_index(name="n")
        .rename(columns={"index": "age"})
    )

    age_groups = raw.copy()
    age_groups["age_groups"] = pd.cut(
        age_groups["age"],
        bins=list(range(0, 81, 10)) + [100],
        right=False,
        labels=[f"{i}-{i + 9}" for i in range(0, 71, 10)] + ["80-100"],
    )

    age_groups = age_groups.groupby("age_groups")["n"].sum().to_frame()
    age_groups["weight"] = age_groups["n"] / age_groups["n"].sum()

    age_groups.to_parquet(produces[0])

    # rki

    age_group_rki = raw.copy()
    age_group_rki["age_group_rki"] = create_age_groups_rki(age_group_rki)

    age_group_rki = age_group_rki.groupby("age_group_rki")["n"].sum().to_frame()
    age_group_rki["weight"] = age_group_rki["n"] / age_group_rki["n"].sum()
    age_group_rki.to_pickle(produces[1])
