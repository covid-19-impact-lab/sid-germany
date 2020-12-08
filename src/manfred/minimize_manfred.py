import hashlib
import itertools
from collections import namedtuple

import numpy as np


def minimize_manfred(
    func,
    x,
    step_sizes,
    max_fun,
    xtol=0.01,
    direction_window=3,
    use_line_search=True,
    line_search_frequency=3,
    n_points_per_line_search=5,
    lower_bounds=None,
    upper_bounds=None,
    max_step_sizes=None,
    default_direct_search_mode="fast",
    convergence_direct_search_mode="thorough",
):
    """Minimize func using the MANFRED algorithm.

    Args:
        func (callable): Python function that takes the argument x
            (a 1d numpy array with parameters) and returns a dictionary
            with the entries "residuals" and "value".
        x (numpy.ndarray): 1d numpy array with parameters.
        initial_step_size (float): The step size in the direct search phase.
        max_fun (int): Maximum number of function evaluations.
        xtol (float): Maximal sum of absolute differences for two
            parameter vectors to be considered equal
        direction_window (int): How many accepted parameters are used to
            determine if a parameter has momentum and we can thus switch
            to one-sided search for that parameter.
        use_line_search (bool): Whether a linesearch is done after each direct
            search step.
        line_search_frequency (int): If use_line_search is true this number
            specifies every how many iterations we do a line search step after
            the direct search step.Line search steps can lead to fast progress
            and/or refined solutions and the number of required function
            evaluations does not depend on the dimensionality. The disadvantage
            is that they make caching more inefficient by leaving the
            grid and that they make it harder to check convergence of the
            direct search with a given step size. 3 seems to be a sweet spot.
        n_points_per_line_search (int): How many points are tried during a line search.
        max_step_sizes (float or list): Maximum step size that can be taken in any
            direction during the line search step. Needs to be a float or a list of
            the same length as step_sizes. A large max_step_size can lead to a fast
            convergence if the approximate gradient approximation is good. This is
            especially helpful at the beginning. Later, a small max_step limits the
            search space for the line search and can thus increase precision.

    """
    bounds = _process_bounds(x, lower_bounds, upper_bounds)
    step_sizes, max_step_sizes = _process_step_sizes(step_sizes, max_step_sizes)

    line_search_info = {
        "n_points": n_points_per_line_search,
    }

    direct_search_info = {
        "direction_window": direction_window,
    }

    state = {
        "func_counter": 0,
        "iter_counter": 0,
        "inner_iter_counter": 0,
        "cache": {hash_array(x): {"x": x, "evals": [func(x)]}},
        "history": [],
    }

    convergence_criteria = {"xtol": xtol, "max_fun": max_fun}

    current_x = x
    for step_size, max_step_size in zip(step_sizes, max_step_sizes):
        state["inner_iter_counter"] = 0
        while not _has_converged(state, convergence_criteria):
            current_x, state = do_manfred_direct_search(
                func=func,
                current_x=current_x,
                step_size=step_size,
                state=state,
                info=direct_search_info,
                bounds=bounds,
                mode=default_direct_search_mode,
            )

            if use_line_search and (state["iter_counter"] % line_search_frequency) == 0:
                current_x, state = do_manfred_line_search(
                    func=func,
                    current_x=current_x,
                    step_size=step_size,
                    state=state,
                    info=line_search_info,
                    bounds=bounds,
                    max_step_size=max_step_size,
                )

            needs_thorough_search = (
                not _x_has_changed(state, convergence_criteria)
                and convergence_direct_search_mode in ("thorough", "very-thorough")
                and _is_below_max_fun(state, convergence_criteria)
            )

            if needs_thorough_search:
                current_x, state = do_manfred_direct_search(
                    func=func,
                    current_x=current_x,
                    step_size=step_size,
                    state=state,
                    info=direct_search_info,
                    bounds=bounds,
                    mode="thorough",
                )

            needs_very_thorough_search = (
                not _x_has_changed(state, convergence_criteria)
                and convergence_direct_search_mode == "very-thorough"
                and _is_below_max_fun(state, convergence_criteria)
            )
            if needs_very_thorough_search:
                current_x, state = do_manfred_direct_search(
                    func=func,
                    current_x=current_x,
                    step_size=step_size,
                    state=state,
                    info=direct_search_info,
                    bounds=bounds,
                    mode="very-thorough",
                )

            state["iter_counter"] = state["iter_counter"] + 1
            state["inner_iter_counter"] = state["inner_iter_counter"] + 1

    out_history = {"criterion": [], "x": []}
    for x_hash in state["history"]:
        cache_entry = state["cache"][x_hash]
        out_history["criterion"].append(_aggregate_evaluations(cache_entry["evals"]))
        out_history["x"].append(cache_entry["x"])

    res = {
        "solution_x": current_x,
        "n_criterion_evaluations": state["func_counter"],
        "n_iterations": state["iter_counter"],
        "history": out_history,
    }

    return res


def _process_bounds(x, lower_bounds, upper_bounds):
    if lower_bounds is None:
        lower_bounds = np.full(len(x), -np.inf)
    if upper_bounds is None:
        upper_bounds = np.full(len(x), np.inf)
    bounds = {"lower": lower_bounds, "upper": upper_bounds}

    if not _is_in_bounds(x, bounds):
        raise ValueError("x must be inside the bounds.")

    return bounds


def _process_step_sizes(step_sizes, max_step_sizes):
    if isinstance(step_sizes, (list, tuple, np.ndarray)):
        step_sizes = list(step_sizes)
    elif isinstance(step_sizes, (float, int)):
        step_sizes = [float(step_sizes)]
    else:
        raise ValueError("step_sizes must be int, float or list thereof.")

    if max_step_sizes is None:
        max_step_sizes = [size * 5 for size in step_sizes]
    elif isinstance(max_step_sizes, (list, tuple, np.ndarray)):
        max_step_sizes = list(max_step_sizes)
        assert len(max_step_sizes) == len(step_sizes)
    elif isinstance(max_step_sizes, (int, float)):
        max_step_sizes = [max_step_sizes] * len(step_sizes)

    for ss, mss in zip(step_sizes, max_step_sizes):
        assert ss <= mss

    return step_sizes, max_step_sizes


def _has_converged(state, convergence_criteria):
    has_changed = _x_has_changed(state, convergence_criteria)
    below_max_fun = _is_below_max_fun(state, convergence_criteria)
    converged = (not has_changed) or (not below_max_fun)
    return converged


def _x_has_changed(state, convergence_criteria):
    if state["inner_iter_counter"] > 0:
        current_x = state["cache"][state["history"][-1]]["x"]
        last_x = state["cache"][state["history"][-2]]["x"]
        has_changed = np.abs(last_x - current_x).max() > convergence_criteria["xtol"]
    else:
        has_changed = True
    return has_changed


def _is_below_max_fun(state, convergence_criteria):
    return state["func_counter"] < convergence_criteria["max_fun"]


def do_manfred_direct_search(func, current_x, step_size, state, info, bounds, mode):
    search_strategies = _determine_search_strategies(current_x, state, info, mode)
    x_sample = _get_direct_search_sample(
        current_x, step_size, search_strategies, bounds
    )

    evaluations, state = _do_evaluations(func, x_sample, state, "aggregated")

    argmin = np.argmin(evaluations)
    next_x = x_sample[argmin]
    state["history"].append(hash_array(next_x))

    return next_x, state


def do_manfred_line_search(
    func, current_x, step_size, state, info, bounds, max_step_size
):
    direction = _calculate_manfred_direction(current_x, step_size, state["cache"])
    x_sample = _get_line_search_sample(
        current_x, direction, info, bounds, max_step_size
    )

    evaluations, state = _do_evaluations(
        func=func,
        x_sample=x_sample,
        state=state,
        return_type="aggregated",
    )

    argmin = np.argmin(evaluations)
    next_x = x_sample[argmin]
    state["history"].append(hash_array(next_x))

    return next_x, state


def _get_line_search_sample(current_x, direction, info, bounds, max_step_size):
    upper_line_search_bound = _find_maximal_line_search_step(
        current_x, direction, bounds, max_step_size
    )
    grid = np.linspace(0, upper_line_search_bound, info["n_points"] + 1)
    x_sample = [current_x + step * direction for step in grid]
    # make absolutely sure the hash of the already evaluated point does not change
    x_sample[0] = current_x
    return x_sample


def _do_evaluations(func, x_sample, state, return_type="aggregated"):
    cache = state["cache"]
    x_hashes = [hash_array(x) for x in x_sample]
    need_to_evaluate = [
        x for x, x_hash in zip(x_sample, x_hashes) if x_hash not in cache
    ]
    new_evaluations = [func(x) for x in need_to_evaluate]
    for x, evaluation in zip(need_to_evaluate, new_evaluations):
        cache = _add_to_cache(x, evaluation, cache)

    all_results = [cache[x_hash]["evals"] for x_hash in x_hashes]

    if return_type == "aggregated":
        all_results = [_aggregate_evaluations(res) for res in all_results]

    state["func_counter"] = state["func_counter"] + len(need_to_evaluate)

    return all_results, state


def _calculate_manfred_direction(current_x, step_size, cache):
    pos_values = _get_values_for_pseudo_gradient(current_x, step_size, 1, cache)
    neg_values = _get_values_for_pseudo_gradient(current_x, step_size, -1, cache)
    f0 = _aggregate_evaluations(cache[hash_array(current_x)]["evals"])

    two_sided_gradient = (pos_values - neg_values) / (2 * step_size)
    right_gradient = (pos_values - f0) / step_size
    left_gradient = (f0 - neg_values) / step_size

    gradient = two_sided_gradient
    gradient = np.where(np.isnan(gradient), right_gradient, gradient)
    gradient = np.where(np.isnan(gradient), left_gradient, gradient)
    gradient = np.where(np.isnan(gradient), 0, gradient)

    direction = -gradient
    return direction


def _get_values_for_pseudo_gradient(current_x, step_size, sign, cache):
    x_hashes = []
    for i, val in enumerate(current_x):
        x = current_x.copy()
        if sign > 0:
            x[i] = val + step_size
        else:
            x[i] = val - step_size
        x_hashes.append(hash_array(x))

    values = []
    for x_hash in x_hashes:
        if x_hash in cache:
            values.append(_aggregate_evaluations(cache[x_hash]["evals"]))
        else:
            values.append(np.nan)
    return np.array(values)


def _determine_search_strategies(current_x, state, info, mode):
    resid_strats = _determine_fast_residual_strategies(current_x, state)
    hist_strats = _determine_fast_history_strategies(current_x, state, info)
    strats = [_combine_strategies(s1, s2) for s1, s2 in zip(resid_strats, hist_strats)]

    if mode == "thorough":
        strats = [s if s != "fixed" else "two-sided" for s in strats]
    elif mode == "very-thorough":
        strats = ["two-sided"] * len(strats)

    return strats


def _determine_fast_residual_strategies(current_x, state):
    x_hash = hash_array(current_x)
    evals = state["cache"][x_hash]["evals"]
    residual_sum = np.sum([evaluation["residuals"] for evaluation in evals])

    if residual_sum > 0:
        strategies = ["left"] * len(current_x)
    else:
        strategies = ["right"] * len(current_x)

    return strategies


def _determine_fast_history_strategies(current_x, state, info):
    effective_window = min(info["direction_window"], len(state["history"]))
    if effective_window >= 2:
        relevant_history = [
            state["cache"][x_hash]["x"]
            for x_hash in state["history"][-effective_window:]
        ]
        diffs = np.diff(relevant_history, axis=0)
        all_zero = (diffs == 0).all(axis=0)
        left = (diffs <= 0).all(axis=0) & ~all_zero
        right = (diffs >= 0).all(axis=0) & ~all_zero

        strategies = _bools_to_strategy(left, right)
    else:
        strategies = ["two-sided"] * len(current_x)
    return strategies


def _get_direct_search_sample(current_x, step_size, search_strategies, bounds):

    strategies = {
        "two-sided": lambda x, step: [x - step, x, x + step],
        "right": lambda x, step: [x, x + step],
        "left": lambda x, step: [x - step, x],
        "fixed": lambda x, step: [x],
    }

    points_per_param = []
    for val, strategy in zip(current_x, search_strategies):
        points = strategies[strategy](val, step_size)
        points_per_param.append(points)

    raw_sample = map(np.array, itertools.product(*points_per_param))
    sample = [x for x in raw_sample if _is_in_bounds(x, bounds)]
    return sample


def _aggregate_evaluations(evaluations):
    return np.mean([evaluation["value"] for evaluation in evaluations])


def _add_to_cache(x, evaluation, cache):
    x_hash = hash_array(x)
    if x_hash in cache:
        cache[x_hash]["evals"].append(evaluation)
    else:
        cache[x_hash] = {"x": x, "evals": [evaluation]}
    return cache


def _bools_to_strategy(left, right):
    strategies = []
    for i in range(len(left)):
        if left[i]:
            strategies.append("left")
        elif right[i]:
            strategies.append("right")
        else:
            strategies.append("two-sided")
    return strategies


def _combine_strategies(s1, s2):
    strategies = {s1, s2}
    if "left" in strategies and "right" in strategies:
        combined = "fixed"
    elif len(strategies) == 1:
        combined = list(strategies)[0]
    else:
        combined = list(strategies - {"two-sided"})[0]
    return combined


def hash_array(arr):
    """Create a hashsum for fast comparison of numpy arrays."""
    # make the array exactly representable as float
    arr = 1 + arr - 1
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _namedtuple_from_dict(dict_, name):
    return namedtuple(name, dict_)(**dict_)


def _is_in_bounds(x, bounds):
    return (x >= bounds["lower"]).all() and (x <= bounds["upper"]).all()


def _find_maximal_line_search_step(x, direction, bounds, max_step_size):

    upper_bounds = np.minimum(x + max_step_size, bounds["upper"])
    lower_bounds = np.maximum(x - max_step_size, bounds["lower"])

    max_steps = []
    for xi, di, lb, ub in zip(x, direction, lower_bounds, upper_bounds):
        max_steps.append(_find_one_max_step(xi, di, lb, ub))
    return min(max_steps)


def _find_one_max_step(xi, di, lb, ub):
    if di == 0:
        max_step = np.inf
    elif di > 0:
        max_step = (ub - xi) / di
    else:
        max_step = (lb - xi) / di
    return max_step
