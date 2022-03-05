import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns
import statsmodels.formula.api as sm

from src.config import BLD


DROSTEN_SENSITIVITIES_BY_CT_BIN = {
    "below_25": 0.955979,
    "25_to_30": 0.454260,
    "above_30": 0.035646,
    "17_to_36": 0.546042,
}


PRODUCES = {
    "plot": BLD
    / "figures"
    / "data"
    / "testing"
    / "sensitivity_params_with_different_methods.pdf",
    "table": BLD / "tables" / "sensitivity_params_with_different_methods.csv",
}


@pytask.mark.produces(PRODUCES)
def task_get_and_plot_sensitivity_params(produces):
    to_concat = []
    for ct_method in ["linear", "lookup"]:
        for sensitivity_method in ["linear", "lookup"]:
            sensitivities = calculate_sensitivities(
                ct_method=ct_method,
                sensitivity_method=sensitivity_method,
            )
            sensitivities.name = f"{ct_method} ct, {sensitivity_method} sensitivity"
            to_concat.append(sensitivities)
    df = pd.concat(to_concat, axis=1)

    df["average"] = df.mean(axis=1)
    df["lower_envelope"] = df.min(axis=1)
    df["old"] = [0.35, 0.35, 0.88] + [0.95] * 8 + [0.5] * 6

    df.to_csv(produces["table"])

    fig, ax = plt.subplots()
    for col in df.columns:
        sns.lineplot(x=-df.index, y=df[col], ax=ax, label=col)

    fig.savefig(produces["plot"])


def calculate_sensitivities(
    cd_infectious_true_grid=None,
    sensitivity_data=DROSTEN_SENSITIVITIES_BY_CT_BIN,
    ct_method="linear",
    sensitivity_method="linear",
):
    """Calculate sensitivities on a grid of values for cd_infectious_true."""
    if cd_infectious_true_grid is None:
        cd_infectious_true_grid = list(range(2, -15, -1))
    sensitivities = []
    for cd_val in cd_infectious_true_grid:
        if ct_method == "linear":
            ct = _calculate_linear_ct(cd_val)
        else:
            ct = _calculate_lookup_ct(cd_val)

        if sensitivity_method == "linear":
            sens = _calculate_linear_sensitivity(ct, sensitivity_data)
        else:
            sens = _calculate_lookup_sensitvity(ct, sensitivity_data)
        sensitivities.append(sens)

    sensitivities = pd.Series(sensitivities, index=cd_infectious_true_grid)
    return sensitivities


def _calculate_lookup_ct(cd_infectious_true):
    """Calculate the mean Ct value using the days since infectiousness started.

    This uses the raw violin plots of Cosentino2021 (Figure 1A).

    """
    days_since_symptoms = -cd_infectious_true - 2
    # sensitivity is only applied to rapid tests of infected individuals
    if -6 < days_since_symptoms < 0:
        # use Jang2020
        ct = 21.4 - 1.2 * days_since_symptoms
    # Cosentino2021
    elif days_since_symptoms in [0, 1]:
        ct = 23
    elif 2 <= days_since_symptoms < 6:
        ct = 26
    elif 6 <= days_since_symptoms < 12:
        # 10 days after symptom onset no one is infectious anymore in sid
        ct = 32
    else:
        ct = 40
    return ct


def _calculate_linear_ct(cd_infectious_true):
    """Calculate the mean Ct value using the days since infectiousness started.

    This uses the linear estimation of Cosentino2021.

    There are 1-5 days from infection to infectiousness.
    If symptoms develop they do so 1-2 days after infectiousness starts.

    Sensitivity is only applied to rapid tests of infected individuals so
    we need not worry about uninfected individuals.

    """
    days_since_symptoms = -cd_infectious_true - 2
    ct = _calculate_linear_ct_from_days_since_symptoms(days_since_symptoms)
    return ct


def _calculate_linear_ct_from_days_since_symptoms(days_since_symptoms):
    """Calculate the mean Ct value from the number of days since symptom onset.

    Negative values are allowed.

    """
    if days_since_symptoms > 0:
        # use Cosentino2021
        ct = 21.4 + 1.03 * days_since_symptoms
    elif days_since_symptoms > -6:
        # use Jang2020
        ct = 21.4 - 1.2 * days_since_symptoms
    else:
        ct = 40
    return ct


def _calculate_lookup_sensitvity(ct, sensitivity_data):
    """Convert Ct value to sensitivity looking up the bin's mean from Drosten."""
    if ct <= 25:
        sensitivity = sensitivity_data["below_25"]
    elif ct < 30:
        sensitivity = sensitivity_data["25_to_30"]
    else:
        sensitivity = sensitivity_data["above_30"]
    return sensitivity


def _calculate_linear_sensitivity(ct, sensitivity_data):
    """Convert Ct value to sensitivity interpolating between Drosten's bin means.

    The interpolation makes the following assumptions:
    - The ct values are uniformly distributed between 17 and 36
    - The sensitivity depends linearly on the ct value

    """
    midpoints = pd.Series(
        {
            "below_25": np.mean([17, 25]),
            "25_to_30": np.mean([25, 30]),
            "above_30": np.mean([30, 36]),
        },
        name="ct",
    )

    sensitivity_data_sr = pd.Series(sensitivity_data, name="sensitivity")

    df = pd.concat([midpoints, sensitivity_data_sr], axis=1)

    res = sm.ols("sensitivity ~ ct", data=df).fit()
    constant, slope = res.params["Intercept"], res.params["ct"]

    interpolated = np.clip(constant + ct * slope, 0, 1)

    return interpolated
