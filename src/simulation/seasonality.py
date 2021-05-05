import numpy as np
import pandas as pd


def seasonality_model(params, dates, seed, contact_models):  # noqa: U100
    df = pd.DataFrame(index=dates)
    weak_effect = params.loc[
        ("seasonality_effect", "seasonality_effect", "weak"), "value"
    ]
    strong_effect = params.loc[
        ("seasonality_effect", "seasonality_effect", "strong"), "value"
    ]
    for name in contact_models:
        effect = strong_effect if "other" in name else weak_effect
        df[name] = create_seasonality_series(dates, effect)
    return df


def create_seasonality_series(dates, season_effect):
    base_sine_curve = np.sin(np.pi * (dates.dayofyear.to_numpy() / 182.5 + 0.5))
    factor_arr = 1 - 0.5 * season_effect + 0.5 * season_effect * base_sine_curve
    factor_sr = pd.Series(factor_arr, index=dates)
    return factor_sr
