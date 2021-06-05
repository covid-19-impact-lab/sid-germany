from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from estimagic import minimize
from scipy.optimize import minimize as scipy_minimize

from src.manfred.minimize_manfred import minimize_manfred
from src.manfred.minimize_manfred_estimagic import minimize_manfred_estimagic


def criterion_function(x, seed, true_x, data, noise_level=0):
    np.random.seed(seed)
    true_y_hat = data @ true_x
    calc_y = data @ x
    noise = (
        ((true_y_hat - calc_y) ** 2 + 1)
        * noise_level
        * np.random.normal(size=len(data))
    )
    true_y = true_y_hat + noise
    base_residuals = true_y - calc_y
    residuals = np.abs(base_residuals * 3) ** 1.5 * np.sign(base_residuals)

    value = residuals @ residuals
    return {"value": value, "root_contributions": residuals}


def scipy_criterion_function(x, true_x, data, noise_level):
    seed = np.random.randint(10000)
    return criterion_function(x, seed, true_x, data, noise_level)["value"]


def estimagic_criterion_function(params, true_x, data, noise_level):  # noqa: U100
    seed = np.random.choice(10000)
    out = criterion_function(
        params["value"].to_numpy(), seed, true_x, data, noise_level
    )
    return out


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
        if col == "criterion":
            sns.lineplot(
                x=np.arange(len(plot_data)) + 1,
                y=np.log(plot_data[col]),
                ax=ax,
            )
        else:
            sns.lineplot(
                x=np.arange(len(plot_data)) + 1,
                y=plot_data[col],
                ax=ax,
            )

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    true_x = np.array([0.08, 0.15, 0.22, 0.31, 0.37, 0.28])
    n_params = len(true_x)
    start_x = np.full(n_params, 0.23)
    lower_bounds = np.zeros(n_params)
    upper_bounds = np.ones(n_params)

    mean = np.zeros(n_params)
    corr = 0.25
    cov = np.eye(n_params) * (1 - corr) + np.ones((n_params, n_params)) * corr
    np.random.seed(1234)
    data = np.random.multivariate_normal(mean, cov, size=500)

    # ==================================================================================
    # Simple test
    # ==================================================================================

    scipy_test_func = partial(
        scipy_criterion_function, true_x=true_x, data=data, noise_level=0
    )
    test_func = partial(criterion_function, true_x=true_x, data=data, noise_level=0)

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
        linesearch_n_points=12,
        gradient_weight=gradient_weight,
    )

    scipy_res = scipy_minimize(
        scipy_test_func, start_x, method="Nelder-Mead", options={"maxfev": 100_000}
    )

    fig = plot_history(res)

    fig.savefig(Path(__file__).resolve().parent / "convergence_plot.pdf")

    print("Noise Free Test:           ")  # noqa: T001
    print("Manfred Solution:     ", res["solution_x"].round(2))  # noqa: T001
    print("True Solution:        ", true_x.round(2))  # noqa: T001
    print("Nelder Mead Solution: ", scipy_res.x.round(2))  # noqa: T001
    print("Manfred n_evals:      ", res["n_criterion_evaluations"])  # noqa: T001
    print("Nelder Mead n_evals:  ", scipy_res.nfev, "\n")  # noqa: T001

    # ==================================================================================
    # Very noisy test
    # ==================================================================================
    noise_level = 0.15
    noisy_test_func = partial(
        criterion_function, true_x=true_x, data=data, noise_level=noise_level
    )

    gradient_weight = 0.4

    res = minimize_manfred(
        func=noisy_test_func,
        x=start_x,
        xtol=0.001,
        step_sizes=[0.1, 0.05, 0.02],
        max_fun=1_000_000,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        linesearch_active=[True, True, True],
        max_step_sizes=[0.3, 0.2, 0.1],
        linesearch_n_points=12,
        n_evaluations_per_x=[50, 90, 120],
        gradient_weight=gradient_weight,
    )

    fig = plot_history(res)

    fig.savefig(Path(__file__).resolve().parent / "very_noisy_convergence_plot.pdf")

    print("Very Noisy Test:           ")  # noqa: T001
    print("Manfred Solution:     ", res["solution_x"].round(2))  # noqa: T001
    print("True Solution:        ", true_x.round(2))  # noqa: T001
    print("Manfred n_evals:      ", res["n_criterion_evaluations"])  # noqa: T001

    # ==================================================================================
    # Noisy test
    # ==================================================================================

    gradient_weight = 0.5
    noise_level = 0.1
    noisy_test_func = partial(
        criterion_function,
        true_x=true_x,
        data=data,
        noise_level=noise_level,
    )

    res = minimize_manfred(
        func=noisy_test_func,
        x=start_x,
        xtol=0.001,
        step_sizes=[0.1, 0.05, 0.02],
        max_fun=1_000_000,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_step_sizes=[0.3, 0.2, 0.2],
        linesearch_n_points=12,
        n_evaluations_per_x=[50, 90, 120],
        gradient_weight=gradient_weight,
        direction_window=2,
    )

    fig = plot_history(res)

    fig.savefig(Path(__file__).resolve().parent / "noisy_convergence_plot.pdf")

    print("Noisy Test:           ")  # noqa: T001
    print("Manfred Solution:     ", res["solution_x"].round(2))  # noqa: T001
    print("True Solution:        ", true_x.round(2))  # noqa: T001
    print("Manfred n_evals:      ", res["n_criterion_evaluations"])  # noqa: T001

    # ==================================================================================
    # Simple test with estimagic interface
    # ==================================================================================

    gradient_weight = 0.6
    noise_level = 0
    params = pd.DataFrame()
    params["value"] = start_x
    params["lower_bound"] = lower_bounds
    params["upper_bound"] = upper_bounds

    estimagic_func = partial(
        estimagic_criterion_function,
        true_x=true_x,
        noise_level=noise_level,
        data=data,
    )

    algo_options = {
        "step_sizes": [0.1, 0.05, 0.0125],
        "max_step_sizes": [1, 0.2, 0.1],
        "linesearch_n_points": 12,
        "gradient_weight": gradient_weight,
        "convergence_relative_params_tolerance": 0.001,
    }

    estimagic_res = minimize(
        criterion=estimagic_func,
        params=params,
        algorithm=minimize_manfred_estimagic,
        algo_options=algo_options,
        logging=False,
    )

    print("Estimagic Test:       ")  # noqa: T001
    print("Manfred Solution:     ", estimagic_res["solution_x"].round(2))  # noqa: T001
    print("True Solution:        ", true_x.round(2))  # noqa: T001
    print(  # noqa: T001
        "Manfred n_evals:      ", estimagic_res["n_criterion_evaluations"]
    )

    # ==================================================================================
    # Noisy test with estimagic interface
    # ==================================================================================

    gradient_weight = 0.5
    noise_level = 0.1
    params = pd.DataFrame()
    params["value"] = start_x
    params["lower_bound"] = lower_bounds
    params["upper_bound"] = upper_bounds

    estimagic_func = partial(
        estimagic_criterion_function,
        true_x=true_x,
        noise_level=noise_level,
        data=data,
    )

    algo_options = {
        "step_sizes": [0.1, 0.05, 0.02],
        "max_step_sizes": [0.3, 0.2, 0.2],
        "linesearch_n_points": 12,
        "gradient_weight": gradient_weight,
        "noise_n_evaluations_per_x": [50, 90, 120],
        "convergence_relative_params_tolerance": 0.001,
        "direction_window": 3,
    }

    estimagic_res = minimize(
        criterion=estimagic_func,
        params=params,
        algorithm=minimize_manfred_estimagic,
        algo_options=algo_options,
        logging=False,
    )

    print("Estimagic Test:       ")  # noqa: T001
    print("Manfred Solution:     ", estimagic_res["solution_x"].round(2))  # noqa: T001
    print("True Solution:        ", true_x.round(2))  # noqa: T001
    print(  # noqa: T001
        "Manfred n_evals:      ", estimagic_res["n_criterion_evaluations"]
    )
