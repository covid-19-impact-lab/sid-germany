import hashlib

import numpy as np


def do_evaluations(
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
        cache = add_to_cache(x, evaluation, cache)

    all_results = [cache[x_hash]["evals"] for x_hash in x_hashes]

    if return_type == "aggregated":
        all_results = [aggregate_evaluations(res) for res in all_results]

    state["func_counter"] = state["func_counter"] + len(need_to_evaluate)

    return all_results, state


def aggregate_evaluations(evaluations):
    res = np.mean([evaluation["value"] for evaluation in evaluations])
    return res


def add_to_cache(x, evaluation, cache):
    x_hash = hash_array(x)
    if x_hash in cache:
        cache[x_hash]["evals"].append(evaluation)
    else:
        cache[x_hash] = {"x": x, "evals": [evaluation]}
    return cache


def hash_array(arr):
    """Create a hashsum for fast comparison of numpy arrays."""
    # make the array exactly representable as float
    arr = 1 + arr - 1
    return hashlib.sha1(arr.tobytes()).hexdigest()


def is_in_bounds(x, bounds):
    return (x >= bounds["lower"]).all() and (x <= bounds["upper"]).all()
