from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

from src.manfred.minimize_manfred import minimize_manfred


def test_func(x, seed):
    np.random.seed(seed)
    value = len(x) * x @ x + x.sum() ** 2
    residuals = x
    return {"residuals": residuals, "value": value}


def scipy_test_func(x):
    return test_func(x, 1234)["value"]


def plot_history(res, x_names=None):
    x_history = res["history"]["x"]
    dim = len(x_history[0])
    if x_names is None:
        x_names = [f"x_{i}" for i in range(dim)]
    plot_data = pd.DataFrame(x_history, columns=x_names)
    plot_data["criterion"] = res["history"]["criterion"]
    plot_data = plot_data[["criterion"] + x_names]

    n_plots = dim + 1
    fig, axes = plt.subplots(nrows=n_plots, figsize=(4, n_plots * 2.5))

    for col, ax in zip(plot_data.columns, axes):
        sns.lineplot(
            x=np.arange(len(plot_data)) + 1,
            y=plot_data[col],
            ax=ax,
        )

    fig.tight_layout()

    return fig


if __name__ == "__main__":

    print("Unconstrained Test")  # noqa
    start_x = np.array([-0.2, 0.6, 0.8, -0.6, 0.6, -0.4])
    res = minimize_manfred(
        func=test_func,
        x=start_x,
        step_sizes=[0.05, 0.01],
        max_fun=1000,
        lower_bounds=-np.ones(len(start_x)),
        upper_bounds=np.ones(len(start_x)),
        max_step_sizes=[1, 0.2],
        n_points_per_line_search=10,
    )
    scipy_res = minimize(scipy_test_func, start_x, method="Nelder-Mead")

    fig = plot_history(res)

    fig.savefig(Path(__file__).resolve().parent / "convergence_plot.png")

    print("Solution:", res["solution_x"].round(3))  # noqa
    print("manfred_n_evaluations:", res["n_criterion_evaluations"])  # noqa
    print("nelder-mead_n_evaluations", scipy_res.nfev)  # noqa
    print("\n")  # noqa

    print("Constrained Test")  # noqa

    start_x = np.array([0.7, 0.8] * 3)
    res = minimize_manfred(
        func=test_func,
        x=start_x,
        step_sizes=[0.05, 0.01],
        max_fun=1000,
        lower_bounds=np.full(len(start_x), 0.2),
        upper_bounds=np.ones(len(start_x)),
        max_step_sizes=[1, 0.2],
        n_points_per_line_search=10,
    )

    fig = plot_history(res)
    fig.savefig(Path(__file__).resolve().parent / "constrained_convergence_plot.png")

    print("Solution:", res["solution_x"].round(3))  # noqa
    print("manfred_n_evaluations:", res["n_criterion_evaluations"])  # noqa
