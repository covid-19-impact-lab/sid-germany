import pandas as pd
import numpy as np


def seasonality_model(params, dates, seed):  # noqa: U100
    season_effect = params.loc[
        ("seasonality_effect", "seasonality_effect", "seasonality_effect"), "value"
    ]
    base_sine_curve = np.sin(np.pi * (dates.dayofyear.to_numpy() / 182.5 + 0.5))
    factor_arr = 1 - 0.5 * season_effect + 0.5 * season_effect * base_sine_curve
    factor_sr = pd.Series(factor_arr, index=dates)
    return factor_sr
