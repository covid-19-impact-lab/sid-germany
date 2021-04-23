import itertools

import numpy as np
from estimagic.batch_evaluators import joblib_batch_evaluator

from src.manfred.direct_search import do_manfred_direct_search
from src.manfred.linesearch import do_manfred_linesearch
from src.manfred.search_direction import calculate_manfred_direction
from src.manfred.shared import aggregate_evaluations
from src.manfred.shared import do_evaluations
from src.manfred.shared import hash_array
from src.manfred.shared import is_in_bounds


def minimize_manfred(
    func,
    x,
    step_sizes,
    lower_bounds=None,
    upper_bounds=None,
    max_fun=100_000,
    convergence_direct_search_mode="fast",
    direction_window=3,
    xtol=0.01,
    linesearch_active=True,
    linesearch_frequency=3,
    linesearch_n_points=5,
    max_step_sizes=None,
    n_evaluations_per_x=1,
    seed=0,
    gradient_weight=0.5,
    momentum=0.05,
    batch_evaluator=joblib_batch_evaluator,
    batch_evaluator_options=None,
):
    """MANFRED algorithm.

    MANFRED stands for Monotone Approximate Noise resistent algorithm For Robust
    optimization without Exact Derivatives

    It combines a very robust direct search step with an efficient line search step
    based on a search direction that is a byproduct of the direct search.

    It is meant for optimization problems that fulfill the following conditions:

    - A low number of parameters (10 is already a lot)
    - Presence of substantial true noise, i.e. the criterion function is stochastic and
      the noise is large enough to introduce local minima.
    - Bounds on all parameters are known
    - Nonlinear least square problems where the residuals are available. Moreover,
      we assume that all parameters influence the residuals positively over the whole
      parameter space.

    Despite being able to handle small local minima introduced by noise, MANFRED is a
    local optimization algorithm. If you need a global solution in the presence of
    multiple minima you need to run it from several starting points.

    MANFRED has the following features:

    - Highly parallelizable: You can scale MANFRED to up to 2 ** n_params *
      n_evaluations_per_x cores for a non parallelized criterion function.
    - Monotone: Only function values that actually lead to an improvement are used.

    Args:
        xtol (float): Maximal change in parameter
            vectors between two iterations to declare convergence.
        convergence_direct_search_mode (str): One of "fast", "thorough". If thorough,
            convergence is only declared if a two sided search for all parameters
            does not yield any improvement.
        max_fun (int): Maximal number of criterion evaluations. This
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
    if batch_evaluator_options is None:
        batch_evaluator_options = {}

    bounds = _process_bounds(x, lower_bounds, upper_bounds)
    step_sizes, max_step_sizes = _process_step_sizes(step_sizes, max_step_sizes)
    n_evaluations_per_x = _process_n_evaluations_per_x(
        n_evaluations_per_x, len(step_sizes)
    )
    linesearch_active = _process_scalar_or_list_arg(linesearch_active, len(step_sizes))
    linesearch_frequency = _process_scalar_or_list_arg(
        linesearch_frequency, len(step_sizes)
    )

    assert 0 <= gradient_weight <= 1

    state = {
        "func_counter": 0,
        "iter_counter": 0,
        "inner_iter_counter": 0,
        "cache": {},
        "x_history": [hash_array(x)],
        "direction_history": [],
        "seed": itertools.count(seed),
    }

    do_evaluations(
        func,
        [x],
        state,
        n_evaluations_per_x[0],
        return_type="aggregated",
        batch_evaluator=batch_evaluator,
        batch_evaluator_options=batch_evaluator_options,
    )

    current_x = x
    last_iteration_x = x + np.inf
    for step_size, max_step_size, n_evals, use_ls, ls_freq in zip(
        step_sizes,
        max_step_sizes,
        n_evaluations_per_x,
        linesearch_active,
        linesearch_frequency,
    ):
        if state["func_counter"] >= max_fun:
            break

        state["inner_iter_counter"] = 0
        while True:
            if state["func_counter"] >= max_fun:
                break

            current_x, state = do_manfred_direct_search(
                func=func,
                current_x=current_x,
                step_size=step_size,
                state=state,
                direction_window=direction_window,
                bounds=bounds,
                mode="fast",
                n_evaluations_per_x=n_evals,
                batch_evaluator=batch_evaluator,
                batch_evaluator_options=batch_evaluator_options,
            )

            if (current_x != last_iteration_x).any():
                state["x_history"].append(hash_array(current_x))
                direction = calculate_manfred_direction(
                    current_x=current_x,
                    step_size=step_size,
                    state=state,
                    gradient_weight=gradient_weight,
                    momentum=momentum,
                )
                state["direction_history"].append(direction)
                after_direct_search_x = current_x

            if state["func_counter"] >= max_fun:
                break

            if use_ls and (state["iter_counter"] % ls_freq) == 0:
                current_x, state = do_manfred_linesearch(
                    func=func,
                    current_x=current_x,
                    direction=direction,
                    state=state,
                    n_points=linesearch_n_points,
                    bounds=bounds,
                    max_step_size=max_step_size,
                    n_evaluations_per_x=n_evals,
                    batch_evaluator=batch_evaluator,
                    batch_evaluator_options=batch_evaluator_options,
                )
                if (current_x != after_direct_search_x).any():
                    state["x_history"].append(hash_array(current_x))

            if state["func_counter"] >= max_fun:
                break

            # if neither the line search nor the first direct search brought any changes
            # try the more extensive line search mode
            if convergence_direct_search_mode == "thorough":
                if (np.abs(current_x - last_iteration_x) <= xtol).all():
                    current_x, state = do_manfred_direct_search(
                        func=func,
                        current_x=current_x,
                        step_size=step_size,
                        state=state,
                        direction_window=direction_window,
                        bounds=bounds,
                        mode="thorough",
                        n_evaluations_per_x=n_evals,
                        batch_evaluator=batch_evaluator,
                        batch_evaluator_options=batch_evaluator_options,
                    )

                    if (current_x != last_iteration_x).any():
                        state["x_history"].append(hash_array(current_x))
                        direction = calculate_manfred_direction(
                            current_x=current_x,
                            step_size=step_size,
                            state=state,
                            gradient_weight=gradient_weight,
                            momentum=momentum,
                        )
                        state["direction_history"].append(direction)

            state["iter_counter"] = state["iter_counter"] + 1
            state["inner_iter_counter"] = state["inner_iter_counter"] + 1
            # cause convergence
            if (np.abs(current_x - last_iteration_x) <= xtol).all():
                break
            last_iteration_x = current_x

    out_history = {"criterion": [], "x": []}
    for x_hash in state["x_history"]:
        cache_entry = state["cache"][x_hash]
        out_history["criterion"].append(aggregate_evaluations(cache_entry["evals"]))
        out_history["x"].append(cache_entry["x"])

    res = {
        "solution_x": current_x,
        "n_criterion_evaluations": state["func_counter"],
        "n_iterations": state["iter_counter"],
        "history": out_history,
    }

    return res


def _process_bounds(x, lower_bounds, upper_bounds):
    for bounds in [lower_bounds, upper_bounds]:
        if not np.isfinite(bounds).all():
            raise ValueError("All bounds need to be finite.")

    if not (lower_bounds < upper_bounds).all():
        raise ValueError("Lower bounds must be strictly smaller than upper bounds.")

    bounds = {"lower": lower_bounds, "upper": upper_bounds}

    if not is_in_bounds(x, bounds):
        raise ValueError("x must be inside the bounds.")

    return bounds


def _process_step_sizes(step_sizes, max_step_sizes):
    if step_sizes is None:
        step_sizes = 0.1
    step_sizes = _process_scalar_or_list_arg(step_sizes)
    if max_step_sizes is None:
        max_step_sizes = [ss * 3 for ss in step_sizes]
    else:
        max_step_sizes = _process_scalar_or_list_arg(max_step_sizes, len(step_sizes))

    for ss, mss in zip(step_sizes, max_step_sizes):
        assert ss <= mss

    return step_sizes, max_step_sizes


def _process_n_evaluations_per_x(n_evaluations_per_x, target_len):
    processed = _process_scalar_or_list_arg(n_evaluations_per_x, target_len)
    for n_evals in processed:
        assert n_evals >= 1
    return processed


def _process_scalar_or_list_arg(arg, target_len=None):
    if isinstance(arg, (list, tuple, np.ndarray)):
        processed = list(arg)
        if target_len is not None:
            assert len(processed) == target_len
    elif target_len is None:
        processed = [arg]
    else:
        processed = [arg] * target_len
    return processed
