import numpy as np

from src.manfred.shared import do_evaluations


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
