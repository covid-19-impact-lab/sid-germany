import itertools

import numpy as np

from src.manfred.shared import do_evaluations
from src.manfred.shared import hash_array
from src.manfred.shared import is_in_bounds


def do_manfred_direct_search(
    func,
    current_x,
    step_size,
    state,
    direction_window,
    bounds,
    mode,
    n_evaluations_per_x,
    batch_evaluator,
    batch_evaluator_options,
):
    search_strategies = _determine_search_strategies(
        current_x, state, direction_window, mode
    )
    x_sample = _get_direct_search_sample(
        current_x, step_size, search_strategies, bounds
    )

    evaluations, state = do_evaluations(
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


def _determine_search_strategies(current_x, state, direction_window, mode):
    if mode == "fast":
        resid_strats = _determine_strategies_from_residuals(current_x, state)
        hist_strats = _determine_strategies_from_x_history(
            current_x, state, direction_window
        )
        strats = [
            _combine_strategies(s1, s2) for s1, s2 in zip(resid_strats, hist_strats)
        ]
    else:
        strats = ["two-sided"] * len(current_x)

    return strats


def _determine_strategies_from_residuals(current_x, state):
    x_hash = hash_array(current_x)
    evals = state["cache"][x_hash]["evals"]
    residuals = np.array([evaluation["root_contributions"] for evaluation in evals])
    residual_sum = residuals.sum()

    if residual_sum > 0:
        strategies = ["left"] * len(current_x)
    else:
        strategies = ["right"] * len(current_x)

    return strategies


def _determine_strategies_from_x_history(current_x, state, direction_window):
    effective_window = min(direction_window, len(state["x_history"]))
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
    sample = [x for x in raw_sample if is_in_bounds(x, bounds)]
    return sample


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
