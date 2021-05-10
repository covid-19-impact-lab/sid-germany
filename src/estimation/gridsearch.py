import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from estimagic.batch_evaluators import joblib_batch_evaluator

warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")


def run_1d_gridsearch(func, params, loc, gridspec, n_seeds, n_cores):
    """Run a grid search over one parameter."""
    seeds = _get_seeds(n_seeds)
    grid = np.linspace(*gridspec)
    n_points = gridspec[-1]
    arguments = []
    for point, seed in itertools.product(grid, seeds):
        p = params.copy(deep=True)
        p.loc[loc, "value"] = point
        arguments.append({"params": p, "seed": seed})

    results = joblib_batch_evaluator(
        func=func,
        arguments=arguments,
        n_cores=n_cores,
        unpack_symbol="**",
    )
    reshaped_results = _reshape_flat_list_2d(results, (n_points, len(seeds)))

    avg_values = []
    for row in reshaped_results:
        values = [res["value"] for res in row]
        avg_values.append(np.mean(values))

    best_index = np.argmin(avg_values)

    if len(grid) <= 3:
        order = 1
    elif len(grid) <= 5:
        order = 2
    else:
        order = 3

    fig = sns.regplot(
        x=np.repeat(grid, len(seeds)),
        y=[res["value"] for res in results],
        order=order,
    )

    return reshaped_results, grid, best_index, fig


def run_2d_gridsearch(
    func,
    params,
    loc1,
    gridspec1,
    loc2,
    gridspec2,
    n_seeds,
    n_cores,
    mask=None,
    names=("x_1", "x_2"),
    plot_type="heatmap",
):
    """Run a grid search over two parameters."""
    # naming: _x refers to loc1, _y to loc2 and z to function values
    names = list(names)

    if mask is None:
        mask = np.full((gridspec1[-1], gridspec2[-1]), True)
    seeds = _get_seeds(n_seeds)
    grid_x = np.linspace(*gridspec1)
    grid_y = np.linspace(*gridspec2)

    reverse_indexer = np.zeros((mask.sum(), 2), dtype=int)
    counter = 0
    for i, j in itertools.product(range(len(grid_x)), range(len(grid_y))):
        if mask[i, j]:
            reverse_indexer[counter] = [i, j]
            counter += 1

    arguments = []
    for i, j in reverse_indexer:
        for seed in seeds:
            p = params.copy(deep=True)
            p.loc[loc1, "value"] = grid_x[i]
            p.loc[loc2, "value"] = grid_y[j]
            arguments.append({"params": p, "seed": seed})

    results = joblib_batch_evaluator(
        func=func,
        arguments=arguments,
        n_cores=n_cores,
        unpack_symbol="**",
    )

    reshaped_results = _reshape_flat_list_2d(results, (mask.sum(), n_seeds))

    avg_values = []
    for row in reshaped_results:
        values = [res["value"] for res in row]
        avg_values.append(np.mean(values))
    avg_values = np.array(avg_values)

    best_index = np.argmin(avg_values)

    filled_z = np.full(mask.shape, np.nan)
    filled_z[mask] = avg_values

    fig, ax = plt.subplots(figsize=(7, 6))

    best_i_x, best_i_y = reverse_indexer[best_index]
    best_x, best_y = grid_x[best_i_x], grid_y[best_i_y]

    if plot_type == "contour":

        ax.contourf(
            grid_x,
            grid_y,
            filled_z.T,
            levels=20,
            cmap="YlOrBr",
        )
        ax.scatter(best_x, best_y, color="red")
        ax.set_title("Contours of Criterion Evaluations")
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
    elif plot_type == "heatmap":
        df = pd.DataFrame(
            data=filled_z,
            index=map(lambda x: str(x.round(3)), grid_x),
            columns=map(lambda x: str(x.round(3)), grid_y),
        )
        sns.heatmap(df, ax=ax, cmap="YlOrBr")

    fig.tight_layout()

    return reshaped_results, reverse_indexer, best_index, fig


def get_mask_around_diagonal(dim, offset=1, flip=True):
    """Get a mask that is true around diagonal or flipped diagonal.

    By flipped diagonal we mean the diagonal that goes from bottom left
    to top right.

    Args:
        dim (int): Dimension of the (square) mask.
        offset (int): How many rows around the diagonal are
            included on each side.
        flip (bool): Whether the standard or flipped diagonal
            is requested.

    Returns:
        mask (np.ndarray)

    """
    mask = np.full((dim, dim), True)
    mask[np.tril_indices(dim, k=-1 - offset)] = False
    mask[np.triu_indices(dim, k=1 + offset)] = False

    if flip:
        mask = np.fliplr(mask)
    return mask


def _get_seeds(n_seeds):
    return [int(1.458745827 * 1e5 * i) for i in range(n_seeds)]


def _reshape_flat_list_2d(flat_list, shape):
    n_rows, n_cols = shape
    assert len(shape) == 2
    assert np.prod(shape) == len(flat_list), f"{np.prod(shape)}: {len(flat_list)}"

    reshaped = []
    entries = iter(flat_list)
    for _r in range(n_rows):
        row = []
        for _c in range(n_cols):
            row.append(next(entries))
        reshaped.append(row)
    return reshaped
