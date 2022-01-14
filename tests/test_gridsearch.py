import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from src.estimation.gridsearch import run_1d_gridsearch
from src.estimation.gridsearch import run_2d_gridsearch


def func1(params, seed, initial_states_path):  # noqa: U100
    return {"value": params["value"] @ params["value"]}


def func2(params, seed, initial_states_path):  # noqa: U100
    return {"value": (params.loc[0, "value"] - 0.1) ** 2}


def test_2d_gridsearch():
    _, grid, best_index, _ = run_2d_gridsearch(
        func=func1,
        params=pd.DataFrame([0, 0], columns=["value"]),
        loc1=[0],
        gridspec1=(-1, 1, 21),
        loc2=[1],
        gridspec2=(-3, 3, 21),
        n_seeds=1,
        n_cores=1,
        initial_states_path=None,
    )

    assert_array_almost_equal(grid[best_index], np.zeros(2))


def test_1d_gridsearch():
    _, grid, best_index, _ = run_1d_gridsearch(
        func=func2,
        params=pd.DataFrame([0], columns=["value"]),
        loc=[0],
        gridspec=(-1, 1, 21),
        n_seeds=1,
        n_cores=1,
        initial_states_path=None,
    )

    assert np.allclose(grid[best_index], 0.1)
