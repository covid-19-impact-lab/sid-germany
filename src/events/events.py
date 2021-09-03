import warnings

import numpy as np
import pandas as pd
from sid.time import get_date

from src.testing.testing_models import get_piecewise_linear_interpolation_for_one_day


def introduce_b117(states, params, seed):
    np.random.seed(seed)
    date = get_date(states)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="indexing past lexsort depth")
        params = params.loc[("events", "b117_cases_per_100_000"), "value"].copy(
            deep=True
        )
    start_date = pd.Timestamp(params.index.min())
    end_date = pd.Timestamp(params.index.max())

    out = (
        pd.Series(index=states.index, dtype=float)
        .astype("category")
        .cat.add_categories(["base_strain", "b117"])
    )

    if start_date <= date <= end_date:
        n_cases_per_hundred_thousand = get_piecewise_linear_interpolation_for_one_day(
            date, params
        )
        n_cases = int(n_cases_per_hundred_thousand * len(states) / 100_000)
        pool = states.index[~states["immune"]]
        sampled = np.random.choice(pool, size=n_cases, replace=False)
        out[sampled] = "b117"
    return out
