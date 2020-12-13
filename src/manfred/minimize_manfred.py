import functools
import hashlib
import itertools
from collections import namedtuple

import numpy as np
from estimagic.batch_evaluators import joblib_batch_evaluator


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
        "use_line_search": linesearch_active,
        "line_search_frequency": linesearch_frequency,
        "n_points_per_line_search": linesearch_n_points,
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
    use_line_search=True,
    line_search_frequency=3,
    n_points_per_line_search=5,
    max_step_sizes=None,
    n_evaluations_per_x=1,
    seed=0,
    gradient_weight=0.5,
    momentum=0.05,
    batch_evaluator=joblib_batch_evaluator,
    batch_evaluator_options=None,
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
        convergence_direct_search_mode (str): Can be fast or thorough. If thorough,
            convergence is only declared if a two sided search for all parameters
            does not yield any improvement.
        n_evaluations_per_x (int): Number of function evaluations per parameter vector.
            For noisy functions this should be set higher than one. Can be an int or
            list with the same length as step_sizes.
        seed (int): Seed for the random number generator. This is used to start a
            seed sequence. Then each function is evaluated with a different seed.
        gradient_weight (float): Weight of the normalized gradient in the calculation
            of the manfred direction.
        momentum (float): The search direction is momentum * past search direction +
            (1 - momentum) * momentum free current search direction. More momentum can
            help to break out of local minima and to average out noise in the
            search direction calculation.

    """
    if batch_evaluator_options is None:
        batch_evaluator_options = {}

    bounds = _process_bounds(x, lower_bounds, upper_bounds)
    step_sizes, max_step_sizes = _process_step_sizes(step_sizes, max_step_sizes)
    n_evaluations_per_x = _process_n_evaluations_per_x(n_evaluations_per_x, step_sizes)
    use_line_search = _process_scalar_or_list_arg(use_line_search, len(step_sizes))
    line_search_frequency = _process_scalar_or_list_arg(
        line_search_frequency, len(step_sizes)
    )

    assert 0 <= gradient_weight <= 1

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
        "cache": {},
        "x_history": [hash_array(x)],
        "direction_history": [],
        "seed": itertools.count(seed),
    }

    _do_evaluations(
        func,
        [x],
        state,
        n_evaluations_per_x[0],
        return_type="aggregated",
        batch_evaluator=batch_evaluator,
        batch_evaluator_options=batch_evaluator_options,
    )

    convergence_criteria = {"xtol": xtol, "max_fun": max_fun}

    current_x = x
    last_iteration_x = x + np.nan
    for step_size, max_step_size, n_evals, use_ls, ls_freq in zip(
        step_sizes,
        max_step_sizes,
        n_evaluations_per_x,
        use_line_search,
        line_search_frequency,
    ):
        state["inner_iter_counter"] = 0
        while not _has_converged(state, convergence_criteria):
            current_x, state = do_manfred_direct_search(
                func=func,
                current_x=current_x,
                step_size=step_size,
                state=state,
                info=direct_search_info,
                bounds=bounds,
                mode="fast",
                n_evaluations_per_x=n_evals,
                batch_evaluator=batch_evaluator,
                batch_evaluator_options=batch_evaluator_options,
            )

            if (current_x != last_iteration_x).any():
                state["x_history"].append(hash_array(current_x))
                direction = _calculate_manfred_direction(
                    current_x=current_x,
                    step_size=step_size,
                    state=state,
                    gradient_weight=gradient_weight,
                    momentum=momentum,
                )
                state["direction_history"].append(direction)
                after_direct_search_x = current_x

            if use_ls and (state["iter_counter"] % ls_freq) == 0:
                current_x, state = do_manfred_line_search(
                    func=func,
                    current_x=current_x,
                    direction=direction,
                    state=state,
                    info=line_search_info,
                    bounds=bounds,
                    max_step_size=max_step_size,
                    n_evaluations_per_x=n_evals,
                    batch_evaluator=batch_evaluator,
                    batch_evaluator_options=batch_evaluator_options,
                )
                if (current_x != after_direct_search_x).any():
                    state["x_history"].append(hash_array(current_x))

            # if neither the line search nor the first direct search brought any changes
            # try the more extensive line search mode
            if convergence_direct_search_mode == "thorough":
                if (current_x == last_iteration_x).all():
                    current_x, state = do_manfred_direct_search(
                        func=func,
                        current_x=current_x,
                        step_size=step_size,
                        state=state,
                        info=direct_search_info,
                        bounds=bounds,
                        mode="thorough",
                        n_evaluations_per_x=n_evals,
                        batch_evaluator=batch_evaluator,
                        batch_evaluator_options=batch_evaluator_options,
                    )

                    if (current_x != last_iteration_x).any():
                        state["x_history"].append(hash_array(current_x))
                        direction = _calculate_manfred_direction(
                            current_x=current_x,
                            step_size=step_size,
                            state=state,
                            gradient_weight=gradient_weight,
                            momentum=momentum,
                        )
                        state["direction_history"].append(direction)

            # cause convergence
            if (current_x == last_iteration_x).all():
                state["x_history"].append(hash_array(current_x))

            state["iter_counter"] = state["iter_counter"] + 1
            state["inner_iter_counter"] = state["inner_iter_counter"] + 1
            last_iteration_x = current_x

    out_history = {"criterion": [], "x": []}
    for x_hash in state["x_history"]:
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


def _process_n_evaluations_per_x(n_evaluations_per_x, step_sizes):
    processed = _process_scalar_or_list_arg(n_evaluations_per_x, len(step_sizes))
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


def _has_converged(state, convergence_criteria):
    has_changed = _x_has_changed(state, convergence_criteria)
    below_max_fun = _is_below_max_fun(state, convergence_criteria)
    converged = (not has_changed) or (not below_max_fun)
    return converged


def _x_has_changed(state, convergence_criteria):
    if state["inner_iter_counter"] > 0:
        current_x = state["cache"][state["x_history"][-1]]["x"]
        last_x = state["cache"][state["x_history"][-2]]["x"]
        has_changed = np.abs(last_x - current_x).max() > convergence_criteria["xtol"]
    else:
        has_changed = True
    return has_changed


def _is_below_max_fun(state, convergence_criteria):
    return state["func_counter"] < convergence_criteria["max_fun"]


def do_manfred_direct_search(
    func,
    current_x,
    step_size,
    state,
    info,
    bounds,
    mode,
    n_evaluations_per_x,
    batch_evaluator,
    batch_evaluator_options,
):
    search_strategies = _determine_search_strategies(current_x, state, info, mode)
    x_sample = _get_direct_search_sample(
        current_x, step_size, search_strategies, bounds
    )

    evaluations, state = _do_evaluations(
        func,
        x_sample,
        state,
        n_evaluations_per_x,
        "aggregated",
        batch_evaluator=batch_evaluator,
        batch_evaluator_options=batch_evaluator_options,
    )

    if evaluations:
        argmin = np.argmin(evaluations)
        next_x = x_sample[argmin]
    else:
        next_x = current_x

    return next_x, state


def do_manfred_line_search(
    func,
    current_x,
    direction,
    state,
    info,
    bounds,
    max_step_size,
    n_evaluations_per_x,
    batch_evaluator,
    batch_evaluator_options,
):
    x_sample = _get_line_search_sample(
        current_x, direction, info, bounds, max_step_size
    )

    evaluations, state = _do_evaluations(
        func=func,
        x_sample=x_sample,
        state=state,
        n_evaluations_per_x=n_evaluations_per_x,
        return_type="aggregated",
        batch_evaluator=batch_evaluator,
        batch_evaluator_options=batch_evaluator_options,
    )

    if evaluations:
        argmin = np.argmin(evaluations)
        next_x = x_sample[argmin]
    else:
        next_x = current_x

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


def _do_evaluations(
    func,
    x_sample,
    state,
    n_evaluations_per_x,
    return_type,
    batch_evaluator,
    batch_evaluator_options,
):
    cache = state["cache"]
    x_hashes = [hash_array(x) for x in x_sample]
    need_to_evaluate = []
    for x, x_hash in zip(x_sample, x_hashes):
        n_evals = n_evaluations_per_x
        if x_hash in cache:
            n_evals = max(0, n_evals - len(cache[x_hash]["evals"]))

        need_to_evaluate += [x] * n_evals

    arguments = [{"x": x, "seed": next(state["seed"])} for x in need_to_evaluate]

    new_evaluations = batch_evaluator(
        func=func,
        arguments=arguments,
        unpack_symbol="**",
        **batch_evaluator_options,
    )

    for x, evaluation in zip(need_to_evaluate, new_evaluations):
        cache = _add_to_cache(x, evaluation, cache)

    all_results = [cache[x_hash]["evals"] for x_hash in x_hashes]

    if return_type == "aggregated":
        all_results = [_aggregate_evaluations(res) for res in all_results]

    state["func_counter"] = state["func_counter"] + len(need_to_evaluate)

    return all_results, state


def _calculate_manfred_direction(
    current_x, step_size, state, gradient_weight, momentum
):
    cache = state["cache"]
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

    gradient_direction = _normalize_direction(-gradient)

    last_x = cache[state["x_history"][-2]]["x"]
    step_direction = _normalize_direction(current_x - last_x)

    direction = (
        gradient_weight * gradient_direction + (1 - gradient_weight) * step_direction
    )

    dir_hist = state["direction_history"]
    if momentum > 0 and len(dir_hist) >= 1:
        direction = momentum * dir_hist[-1] + (1 - momentum) * direction

    return direction


def _normalize_direction(direction):
    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm
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
    if mode == "fast":
        resid_strats = _determine_fast_strategies_from_residuals(current_x, state)
        hist_strats = _determine_fast_strategies_from_x_history(current_x, state, info)
        strats = [
            _combine_strategies(s1, s2) for s1, s2 in zip(resid_strats, hist_strats)
        ]
    else:
        strats = ["two-sided"] * len(current_x)

    return strats


def _determine_fast_strategies_from_residuals(current_x, state):
    x_hash = hash_array(current_x)
    evals = state["cache"][x_hash]["evals"]
    residuals = np.array([evaluation["root_contributions"] for evaluation in evals])
    residual_sum = residuals.sum()
    residual_std = residuals.std()

    cutoff = 0.15
    if residual_sum > cutoff * residual_std:
        strategies = ["left"] * len(current_x)
    elif residual_sum < -cutoff * residual_std:
        strategies = ["right"] * len(current_x)
    else:
        strategies = ["two-sided"] * len(current_x)

    return strategies


def _determine_fast_strategies_from_x_history(current_x, state, info):
    effective_window = min(info["direction_window"], len(state["x_history"]))
    if effective_window >= 2:
        relevant_x_history = [
            state["cache"][x_hash]["x"]
            for x_hash in state["x_history"][-effective_window:]
        ]
        diffs = np.diff(relevant_x_history, axis=0)
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
    }

    points_per_param = []
    for val, strategy in zip(current_x, search_strategies):
        points = strategies[strategy](val, step_size)
        points_per_param.append(points)

    raw_sample = map(np.array, itertools.product(*points_per_param))
    sample = [x for x in raw_sample if _is_in_bounds(x, bounds)]
    return sample


def _aggregate_evaluations(evaluations):
    res = np.mean([evaluation["value"] for evaluation in evaluations])
    return res


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


def _combine_strategies(resid, hist):
    strategies = {resid, hist}
    if len(strategies) == 1:
        combined = list(strategies)[0]
    elif "two-sided" in strategies:
        combined = list(strategies - {"two-sided"})[0]
    else:
        combined = hist

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
