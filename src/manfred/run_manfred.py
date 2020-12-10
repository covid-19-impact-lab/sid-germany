from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

from src.manfred.minimize_manfred import minimize_manfred


def criterion_function(x, seed, true_x, noise_level):
    np.random.seed(seed)
    sf_sum = 10
    sf_ind = 20
    scaled_sum_diff = (x.sum() - true_x.sum()) * sf_sum
    scaled_individual_diff = (x - true_x) * sf_ind
    poly = np.abs(scaled_sum_diff ** 3) + (np.abs(scaled_individual_diff ** 3)).sum()
    clean = 50 + poly
    noisy = clean + noise_level * clean * np.random.normal()
    return {"value": noisy, "residuals": x - true_x}


def scipy_criterion_function(x, true_x, noise_level):
    seed = np.random.randint(10000)
    return criterion_function(x, seed, true_x, noise_level)["value"]


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
            y=np.log(plot_data[col]),
            ax=ax,
        )

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    true_x = np.array([0.08, 0.15, 0.22, 0.31, 0.37, 0.28])
    n_params = len(true_x)
    start_x = np.full(n_params, 0.725)
    lower_bounds = np.zeros(n_params)
    upper_bounds = np.ones(n_params)

    scipy_test_func = partial(scipy_criterion_function, true_x=true_x, noise_level=0)
    test_func = partial(criterion_function, true_x=true_x, noise_level=0)

    gradient_weight = 0.6

    res = minimize_manfred(
        func=test_func,
        x=start_x,
        xtol=0.001,
        step_sizes=[0.1, 0.05, 0.0125],
        max_fun=10_000,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_step_sizes=[1, 0.2, 0.1],
        n_points_per_line_search=10,
        convergence_direct_search_mode="fast",
        gradient_weight=gradient_weight,
    )

    scipy_res = minimize(
        scipy_test_func, start_x, method="Nelder-Mead", options={"maxfev": 100_000}
    )

    fig = plot_history(res)

    fig.savefig(Path(__file__).resolve().parent / "convergence_plot.png")

    print("Noise Free Test:           ")  # noqa
    print("Manfred Solution:     ", res["solution_x"].round(2))  # noqa
    print("True Solution:        ", true_x.round(2))  # noqa
    print("Nelder Mead Solution: ", scipy_res.x.round(2))  # noqa
    print("Manfred n_evals:      ", res["n_criterion_evaluations"])  # noqa
    print("Nelder Mead n_evals:  ", scipy_res.nfev, "\n")  # noqa

    noise_level = 0.15
    noisy_test_func = partial(
        criterion_function, true_x=true_x, noise_level=noise_level
    )

    res = minimize_manfred(
        func=noisy_test_func,
        x=start_x,
        xtol=0.001,
        step_sizes=[0.1, 0.05, 0.02],
        max_fun=1_000_000,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_step_sizes=[0.3, 0.2, 0.1],
        n_points_per_line_search=12,
        n_evaluations_per_x=[60, 90, 120],
        gradient_weight=gradient_weight,
    )

    fig = plot_history(res)

    fig.savefig(Path(__file__).resolve().parent / "very_noisy_convergence_plot.png")

    print("Very Noisy Test:           ")  # noqa
    print("Manfred Solution:     ", res["solution_x"].round(2))  # noqa
    print("True Solution:        ", true_x.round(2))  # noqa
    print("Manfred n_evals:      ", res["n_criterion_evaluations"])  # noqa

    noise_level = 0.1
    noisy_test_func = partial(
        criterion_function, true_x=true_x, noise_level=noise_level
    )

    res = minimize_manfred(
        func=noisy_test_func,
        x=start_x,
        xtol=0.001,
        step_sizes=[0.1, 0.05, 0.01],
        max_fun=1_000_000,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_step_sizes=[0.3, 0.2, 0.2],
        n_points_per_line_search=12,
        n_evaluations_per_x=[30, 50, 50],
        gradient_weight=gradient_weight,
    )

    fig = plot_history(res)

    fig.savefig(Path(__file__).resolve().parent / "noisy_convergence_plot.png")

    print("Noisy Test:           ")  # noqa
    print("Manfred Solution:     ", res["solution_x"].round(2))  # noqa
    print("True Solution:        ", true_x.round(2))  # noqa
    print("Manfred n_evals:      ", res["n_criterion_evaluations"])  # noqa
