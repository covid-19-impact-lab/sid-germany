import functools

import numpy as np
from estimagic.batch_evaluators import joblib_batch_evaluator

from src.manfred.minimize_manfred import minimize_manfred


def minimize_manfred_estimagic(
    internal_criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    convergence_relative_params_tolerance=0.001,
    convergence_direct_search_mode="fast",
    max_criterion_evaluations=100_000,
    step_sizes=None,
    max_step_sizes=None,
    direction_window=3,
    gradient_weight=0.5,
    momentum=0.05,
    linesearch_active=True,
    linesearch_frequency=3,
    linesearch_n_points=5,
    noise_seed=0,
    noise_n_evaluations_per_x=1,
    batch_evaluator=joblib_batch_evaluator,
    batch_evaluator_options=None,
):
    """MANFRED algorithm with internal estimagic optimizer interface.

    MANFRED stands for Monotone Approximate Noise resistent algorithm For Robust
    optimization without Exact Derivatives

    It combines a very robust direct search step with an efficient line search step
    based on a search direction that is a byproduct of the direct search.

    It is meant for optimization problems that fulfill the following conditions:

    - A low number of parameters (10 is already a lot)
    - Presence of substantial true noise, i.e. the criterion function is stochastic and
      the noise is large enough to introduce local minima.
    - Bounds on all parameters are known

    Despite being able to handle small local minima introduced by noise, MANFRED is a
    local optimization algorithm. If you need a global solution in the presence of
    multiple minima you need to run it from several starting points.

    MANFRED has the following features:

    - Highly parallelizable: You can scale MANFRED to up to 2 ** n_params *
      n_evaluations_per_x cores for a non parallelized criterion function.
    - Monotone: Only function values that actually lead to an improvement are used.

    Args:
        convergence_relative_params_tolerance (float): Maximal change in parameter
            vectors between two iterations to declare convergence.
        convergence_direct_search_mode (str): One of "fast", "thorough". If thorough,
            convergence is only declared if a two sided search for all parameters
            does not yield any improvement.
        max_criterion_evaluations (int): Maximal number of criterion evaluations. This
            The actual number of evaluations might be higher, because we only check
            at the start of each iteration if the maximum is reached.
        step_sizes (float or list): Step size or list of step sizes for the direct
            search step of the optimization. This step size refers to a rescaled
            parameter vector where all lower bounds are 0 and all upper bounds are 1. It
            is thus a relative step size.
            This is also the step size used to calculate an approximated gradient via
            finite differences because the approximated gradient is a free by product
            of the direct search. If a list of step sizes is provided, the algorithm is
            run with each step size in the list until convergence. Especially for noisy
            problems it is good to use a list of decreasing step sizes.
        max_step_sizes (float or list): Maximum step size that can be taken in any
            direction during the line search step. This step size also refers to the
            rescaled parameter vector. It needs to be a float or a list of
            the same length as step_sizes. A large max_step_size can lead to a fast
            convergence if the search direction is good. This is especially helpful at
            the beginning. Later, a small max_step limits the search space for the line
            search and can thus increase precision.
        direction_window (int): How many accepted parameters are used to
            determine if a parameter has momentum and we can thus switch
            to one-sided search for that parameter.
        gradient_weight (float): The search direction for the line search step is a
            weighted average of the negative gradient and direction taken in the last
            successful direct search step (both normalized to unit length).
            gradient_weight determines the weight of the gradient in this combination.
            Since the gradient contains quantitative information on the steepness in
            each direction it can lead to very fast convergence but is more sensitive
            to noise. Moreover, it only contains first order information.
            The direction from the direct search contains some second order information
            and is more robust to noise because it only uses the ordering of function
            values and not their size.
        momentum (float): The search direction is momentum * past search direction +
            (1 - momentum) * momentum free current search direction. More momentum can
            help to break out of local minima and to average out noise in the
            search direction calculation.
        linesearch_active (bool): Whether line search is used.
        linesearch_frequency (int or list): If linesearch_active is True this number
            specifies every how many iterations we do a line search step after
            the direct search step.Line search steps can lead to fast progress
            and/or refined solutions and the number of required function
            evaluations does not depend on the dimensionality. The disadvantage
            is that they make caching more inefficient by leaving the
            grid and that they make it harder to check convergence of the
            direct search with a given step size. 3 seems to be a sweet spot. Can be a
            list with the same length as step_sizes.
        linesearch_n_points (int): At how many points the function is evaluated during
            the line search. More points mean higher precision but also more sensitivity
            to noise.
        noise_seed (int): Starting point of a seed sequence.
        noise_n_evaluations_per_x (int): How often the criterion function is evaluated
            at each parameter vector in order to average out noise.
        batch_evaluator (callable): An estimagic batch evaluator.
        batch_evaluator_options (dict): Keyword arguments for the batch evaluator.

    """
    algo_info = {
        "primary_criterion_entry": "root_contributions",
        "parallelizes": True,
        "needs_scaling": False,
        "name": "manfred",
    }
    if batch_evaluator_options is None:
        batch_evaluator_options = {}

    criterion = functools.partial(
        internal_criterion_and_derivative, algorithm_info=algo_info, task="criterion"
    )

    options = {
        "step_sizes": step_sizes,
        "max_fun": max_criterion_evaluations,
        "convergence_direct_search_mode": convergence_direct_search_mode,
        "xtol": convergence_relative_params_tolerance,
        "direction_window": direction_window,
        "xtol": convergence_relative_params_tolerance,
        "linesearch_active": linesearch_active,
        "linesearch_frequency": linesearch_frequency,
        "linesearch_n_points": linesearch_n_points,
        "max_step_sizes": max_step_sizes,
        "n_evaluations_per_x": noise_n_evaluations_per_x,
        "seed": noise_seed,
        "gradient_weight": gradient_weight,
        "momentum": momentum,
        "batch_evaluator": batch_evaluator,
        "batch_evaluator_options": batch_evaluator_options,
    }

    unit_x = _x_to_unit_cube(x, lower_bounds, upper_bounds)

    def func(x, seed, lower_bounds, upper_bounds):
        x = _x_from_unit_cube(x, lower_bounds, upper_bounds)
        np.random.seed(seed)
        residuals = criterion(x)
        return {"root_contributions": residuals, "value": residuals @ residuals}

    partialed_func = functools.partial(
        func, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )

    res = minimize_manfred(
        func=partialed_func,
        x=unit_x,
        lower_bounds=np.zeros(len(x)),
        upper_bounds=np.ones(len(x)),
        **options,
    )
    return res


def _x_to_unit_cube(x, lower_bounds, upper_bounds):
    return (x - lower_bounds) / (upper_bounds - lower_bounds)


def _x_from_unit_cube(unit_x, lower_bounds, upper_bounds):
    return unit_x * (upper_bounds - lower_bounds) + lower_bounds
