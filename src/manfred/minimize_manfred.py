import hashlib
import itertools

import numpy as np
from scipy import stats


def minimize_manfred(
    func,
    x,
    step_size,
    max_fun,
    xtol=None,
    one_sided_confidence_level=0.5,
    momentum_window=3,
    use_line_search=True,
    line_search_frequency=3,
    relative_line_search_bounds=(0.1, 4),
    n_points_per_line_search=5,
):
    """Minimize func using the MANFRED algorithm.

    Args:
        func (callable): Python function that takes the argument x
            (a 1d numpy array with parameters) and returns a dictionary
            with the entries "residuals" and "value".
        x (numpy.ndarray): 1d numpy array with parameters.
        step_size (float): The step size in the direct search phase.
        max_fun (int): Maximum number of function evaluations.
        xtol (float): Maximal sum of absolute differences for two
            parameter vectors to be considered equal
        one_sided_confidence_level (float): Confidence level for a one sided
            approximate hypothesis test that the residual sum is positive
            or negative. This only holds asympotically, i.e. for a large
            number of residuals. If the sign of the residual sum is clearly
            determined, we switch to one sided search for one iteration.
        momentum_window (int): How many accepted parameters are used to
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
        relative_line_search_bounds (float): lower and upper bound for the line
            search step size, relative to normal step size.
        n_points_per_line_search (int): How many points are tried during a line search.

    """
    if xtol is None:
        xtol = 0.1 * step_size
    func_counter = 0
    current_x = x
    last_x = x + 1
    history = []
    x_hash = hash_array(current_x)
    cache = {x_hash: {"x": current_x, "evals": [func(current_x)]}}
    n_iter = 0

    while func_counter <= max_fun and np.abs(last_x - current_x).max() > xtol:
        last_x = current_x
        current_x, cache, history, func_counter = do_manfred_direct_search(
            func=func,
            current_x=current_x,
            step_size=step_size,
            cache=cache,
            history=history,
            func_counter=func_counter,
            one_sided_confidence_level=one_sided_confidence_level,
            momentum_window=momentum_window,
        )

        if use_line_search and (n_iter % line_search_frequency) == 0:
            current_x, cache, history, func_counter = do_manfred_line_search(
                func=func,
                current_x=current_x,
                step_size=step_size,
                cache=cache,
                relative_line_search_bounds=relative_line_search_bounds,
                n_points_per_line_search=n_points_per_line_search,
                history=history,
                func_counter=func_counter,
            )
        n_iter += 1

    out_history = {"criterion": [], "x": []}
    for x_hash in history:
        cache_entry = cache[x_hash]
        out_history["criterion"].append(_aggregate_evaluations(cache_entry["evals"]))
        out_history["x"].append(cache_entry["x"])

    res = {
        "solution_x": current_x,
        "n_criterion_evaluations": func_counter,
        "n_iterations": n_iter,
        "history": out_history,
    }

    return res


def do_manfred_direct_search(
    func,
    current_x,
    step_size,
    cache,
    history,
    func_counter,
    one_sided_confidence_level,
    momentum_window,
):
    search_strategies = _determine_search_strategies(
        current_x, cache, one_sided_confidence_level, history, momentum_window
    )
    x_sample = _get_direct_search_sample(current_x, step_size, search_strategies)

    evaluations, cache, func_counter = _do_evaluations(
        func=func,
        x_sample=x_sample,
        cache=cache,
        func_counter=func_counter,
        return_type="aggregated",
    )

    argmin = np.argmin(evaluations)
    next_x = x_sample[argmin]
    history = history + [hash_array(next_x)]

    return next_x, cache, history, func_counter


def do_manfred_line_search(
    func,
    current_x,
    step_size,
    cache,
    relative_line_search_bounds,
    n_points_per_line_search,
    history,
    func_counter,
):
    direction = _calculate_manfred_direction(current_x, step_size, cache)
    lower, upper = np.array(relative_line_search_bounds) * step_size

    step_grid = np.linspace(lower, upper, n_points_per_line_search)

    x_sample = [current_x] + [current_x + step * direction for step in step_grid]

    evaluations, cache, func_counter = _do_evaluations(
        func=func,
        x_sample=x_sample,
        cache=cache,
        func_counter=func_counter,
        return_type="aggregated",
    )

    argmin = np.argmin(evaluations)
    next_x = x_sample[argmin]
    history = history + [hash_array(next_x)]

    return next_x, cache, history, func_counter


def _do_evaluations(func, x_sample, cache, func_counter, return_type="aggregated"):
    x_sample_hashes = [hash_array(x) for x in x_sample]
    need_to_evaluate = [
        x for x, x_hash in zip(x_sample, x_sample_hashes) if x_hash not in cache
    ]
    new_evaluations = [func(x) for x in need_to_evaluate]
    for x, evaluation in zip(need_to_evaluate, new_evaluations):
        cache = _add_to_cache(x, evaluation, cache)

    all_results = [cache[x_hash]["evals"] for x_hash in x_sample_hashes]

    if return_type == "aggregated":
        all_results = [_aggregate_evaluations(res) for res in all_results]

    func_counter += len(need_to_evaluate)

    return all_results, cache, func_counter


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


def _determine_search_strategies(
    current_x, cache, one_sided_confidence_level, history, momentum_window
):
    x_hash = hash_array(current_x)
    evals = cache[x_hash]["evals"]
    residuals = np.array([evaluation["residuals"] for evaluation in evals])
    residual_mean = residuals.mean()
    residual_std = residuals.std()

    test_statistic = np.sqrt(len(residuals)) * residual_mean / residual_std
    critical_value = stats.norm.ppf(one_sided_confidence_level)

    if test_statistic > critical_value:
        residual_strategies = ["left"] * len(current_x)
    elif test_statistic < -critical_value:
        residual_strategies = ["right"] * len(current_x)
    else:
        residual_strategies = ["two-sided"] * len(current_x)

    effective_window = min(momentum_window, len(history))
    if effective_window >= 2:
        momentum_history = [
            cache[x_hash]["x"] for x_hash in history[-effective_window:]
        ]
        diffs = np.diff(momentum_history, axis=0)
        all_zero = (diffs == 0).all(axis=0)
        left = (diffs <= 0).all(axis=0) & ~all_zero
        right = (diffs >= 0).all(axis=0) & ~all_zero

        momentum_strategies = _bools_to_strategy(left, right)
    else:
        momentum_strategies = ["two-sided"] * len(current_x)

    strategies = [
        _combine_strategies(s1, s2)
        for s1, s2 in zip(residual_strategies, momentum_strategies)
    ]

    return strategies


def _get_direct_search_sample(current_x, step_size, search_strategies):

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

    return list(map(np.array, itertools.product(*points_per_param)))


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
