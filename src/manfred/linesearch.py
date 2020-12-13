import numpy as np

from src.manfred.shared import aggregate_evaluations
from src.manfred.shared import do_evaluations
from src.manfred.shared import hash_array


def do_manfred_linesearch(
    func,
    current_x,
    direction,
    state,
    n_points,
    bounds,
    max_step_size,
    n_evaluations_per_x,
    batch_evaluator,
    batch_evaluator_options,
):
    x_sample = _get_linesearch_sample(
        current_x, direction, n_points, bounds, max_step_size
    )

    evaluations, state = do_evaluations(
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


def calculate_manfred_direction(current_x, step_size, state, gradient_weight, momentum):
    cache = state["cache"]
    pos_values = _get_values_for_pseudo_gradient(current_x, step_size, 1, cache)
    neg_values = _get_values_for_pseudo_gradient(current_x, step_size, -1, cache)
    f0 = aggregate_evaluations(cache[hash_array(current_x)]["evals"])

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


def _get_linesearch_sample(current_x, direction, n_points, bounds, max_step_size):
    upper_linesearch_bound = _find_maximal_linesearch_step(
        current_x, direction, bounds, max_step_size
    )
    grid = np.linspace(0, upper_linesearch_bound, n_points + 1)
    x_sample = [current_x + step * direction for step in grid]
    # make absolutely sure the hash of the already evaluated point does not change
    x_sample[0] = current_x
    return x_sample


def _find_maximal_linesearch_step(x, direction, bounds, max_step_size):

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
            values.append(aggregate_evaluations(cache[x_hash]["evals"]))
        else:
            values.append(np.nan)
    return np.array(values)
