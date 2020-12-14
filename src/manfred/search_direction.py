import numpy as np

from src.manfred.shared import aggregate_evaluations
from src.manfred.shared import hash_array


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
