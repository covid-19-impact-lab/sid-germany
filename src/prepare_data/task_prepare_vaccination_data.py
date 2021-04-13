import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sid.colors import get_colors

from src.config import BLD
from src.config import POPULATION_GERMANY
from src.simulation.plotting import style_plot


plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

OUT_PATH = BLD / "data" / "vaccinations"


@pytask.mark.depends_on(BLD / "data" / "raw_time_series" / "vaccinations.xlsx")
@pytask.mark.produces(
    {
        "vaccination_shares_exp": OUT_PATH / "vaccination_shares_exp.pkl",
        "vaccination_shares_quadratic": OUT_PATH / "vaccination_shares_quadratic.pkl",
        "vaccination_shares_raw": OUT_PATH / "vaccination_shares_raw.pkl",
        "fig_first_dose": OUT_PATH / "first_dose.png",
        "fig_vaccination_shares": OUT_PATH / "vaccination_shares.png",
        "fig_fit_exp": OUT_PATH / "fitness_prediction_exp.png",
        "fig_fit_quadratic": OUT_PATH / "fitness_prediction_quadratic.png",
    }
)
def task_prepare_vaccination_data(depends_on, produces):
    df = pd.read_excel(depends_on, sheet_name="Impfungen_proTag")
    df = _clean_vaccination_data(df)
    # this is for comparing with newspaper sites
    fig, ax = _plot_series(df["share_with_first_dose"], "Share with 1st Dose")
    fig.savefig(produces["fig_first_dose"], dpi=200, transparent=False, facecolor="w")
    plt.close()

    vaccination_shares = df["share_with_first_dose"].diff().dropna()
    vaccination_shares.to_pickle(produces["vaccination_shares_raw"])

    # the first individuals to be vaccinated were nursing homes which are not
    # in our synthetic data so we exclude the first 1% of vaccinations to
    # be going to them.
    vaccination_shares[vaccination_shares.cumsum() <= 0.01] = 0

    # because of strong weekend effects we smooth and extrapolate into the future
    smoothed = vaccination_shares.rolling(7, min_periods=1).mean().dropna()

    fitted_exp, prediction_exp = _get_exponential_vaccination_prediction(
        vaccination_shares
    )
    fig, ax = fitness_plot(vaccination_shares, smoothed, fitted_exp)
    fig.savefig(produces["fig_fit_exp"], dpi=200, transparent=False, facecolor="w")
    plt.close()

    fitted_quadratic, prediction_quadratic = _get_quadratic_vaccination_prediction(
        vaccination_shares
    )
    fig, ax = fitness_plot(vaccination_shares, smoothed, fitted_quadratic)
    fig.savefig(
        produces["fig_fit_quadratic"], dpi=200, transparent=False, facecolor="w"
    )
    plt.close()

    start_date = smoothed.index.min() - pd.Timedelta(days=1)
    past = pd.Series(data=0, index=pd.date_range("2020-01-01", start_date))

    expanded_exp = pd.concat([past, smoothed, prediction_exp]).sort_index()
    _test_expanded(expanded_exp)
    expanded_exp.to_pickle(produces["vaccination_shares_exp"])

    expanded_quadratic = pd.concat([past, smoothed, prediction_quadratic]).sort_index()
    _test_expanded(expanded_quadratic)
    expanded_quadratic.to_pickle(produces["vaccination_shares_quadratic"])

    labeled = [
        ("raw data", vaccination_shares),
        ("smoothed", smoothed),
        ("fitted (exponential)", fitted_exp),
        ("prediction (exponential)", prediction_exp[:"2021-05-01"]),
        ("fitted (quadratic)", fitted_quadratic),
        ("prediction (quadratic)", prediction_quadratic[:"2021-05-01"]),
    ]
    fig, ax = _plot_labeled_series(labeled)
    fig.savefig(
        produces["fig_vaccination_shares"], dpi=200, transparent=False, facecolor="w"
    )
    plt.close()


def _clean_vaccination_data(df):
    # drop last two rows (empty and total vaccinations)
    df = df[df["Datum"].isnull().cumsum() == 0].copy(deep=True)
    df["date"] = pd.to_datetime(df["Datum"], format="%m/%d/%yyyy")
    # check date conversion was correct
    assert df["date"].min() == pd.Timestamp(year=2020, month=12, day=27)
    df = df.set_index("date")
    df["received_first_dose"] = df["Einmal geimpft"].cumsum()
    df["share_with_first_dose"] = df["received_first_dose"] / POPULATION_GERMANY
    return df


def _get_exponential_vaccination_prediction(data):
    """Predict the vaccination data into the future using log data."""
    to_use = data["2021-02-01":"2021-03-14"]
    # stop there because of AstraZeneca stop
    y = np.log(to_use)
    exog = pd.DataFrame(index=y.index)
    exog["constant"] = 1
    exog["days_since_march"] = _get_days_since_march_first(exog)

    model = sm.OLS(endog=y, exog=exog)
    results = model.fit()
    fitted = np.exp(exog.dot(results.params))

    start_date = data.index[-1] + pd.Timedelta(days=1)
    start_point = np.log(data[-1])
    log_daily_increase = results.params["days_since_march"]
    days_to_extrapolate = 56  # 8 weeks
    log_prediction = start_point + np.arange(days_to_extrapolate) * log_daily_increase
    dates = pd.date_range(
        start=start_date,
        end=start_date + pd.Timedelta(days=days_to_extrapolate - 1),
    )
    prediction = np.exp(log_prediction)
    prediction = pd.Series(prediction, index=dates)

    return fitted, prediction


def _get_quadratic_vaccination_prediction(data):
    """Predict the vaccination data into the future using a parabola."""
    ols_data = data["2021-01-15":"2021-03-14"].to_frame()
    # stop there because of AstraZeneca stop
    ols_data["days_since_march"] = _get_days_since_march_first(ols_data)

    model = smf.ols(
        "share_with_first_dose ~ days_since_march + np.power(days_since_march, 2)",
        data=ols_data,
    )
    results = model.fit()
    fitted = results.predict(ols_data)

    start_date = data.index[-1] + pd.Timedelta(days=1)
    days_to_extrapolate = 56  # 8 weeks
    dates = pd.date_range(
        start=start_date,
        end=start_date + pd.Timedelta(days=days_to_extrapolate - 1),
    )
    future_x = pd.DataFrame(index=dates)
    future_x["days_since_march"] = _get_days_since_march_first(future_x)
    prediction = results.predict(future_x)
    point_to_start_at = data[start_date - pd.Timedelta(days=1)]
    diff_to_abstract = prediction[0] - point_to_start_at
    prediction = prediction - diff_to_abstract

    return fitted, prediction


def _get_days_since_march_first(df):
    """Get the number of days since March 1st from a date index."""
    return (df.index - pd.Timestamp("2021-03-01")).days


def fitness_plot(actual, smoothed, fitted):
    """Compare the actual, smoothed and fitted share becoming immune."""
    colors = get_colors("categorical", 4)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        x=actual.index,
        y=actual,
        label="actual data",
        linewidth=2,
        color=colors[0],
    )
    sns.lineplot(
        x=smoothed.index, y=smoothed, label="smoothed", linewidth=2, color=colors[1]
    )
    sns.lineplot(x=fitted.index, y=fitted, label="fitted", linewidth=2, color=colors[3])
    ax.set_title("Fitness Plot")
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax


def _plot_series(sr, title, label=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=sr.index, y=sr, label=label)
    ax.set_title(title)
    fig, ax = style_plot(fig, ax)
    fig.tight_layout()
    return fig, ax


def _plot_labeled_series(labeled):
    title = "Actual and Extrapolated Share Receiving the Vaccination"
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = get_colors("categorical", len(labeled))
    for (label, sr), color in zip(labeled, colors):
        sns.lineplot(
            x=sr.index,
            y=sr,
            label=label,
            linewidth=2,
            color=color,
        )
    fig, ax = style_plot(fig, ax)
    ax.set_title(title)
    ax.set_ylabel("")
    fig.tight_layout()
    return fig, ax


def _test_expanded(sr):
    assert sr.index.is_monotonic, "index is not monotonic."
    assert not sr.index.duplicated().any(), "Duplicate dates in Series."
    assert (sr.index == pd.date_range(start=sr.index.min(), end=sr.index.max())).all()
