import numpy as np
from numpy.testing import assert_array_almost_equal

from src.manfred.direct_search import _get_direct_search_sample


def test_get_direct_search_sample():
    calculated = _get_direct_search_sample(
        current_x=np.array([0.2, 0.5]),
        step_size=0.05,
        search_strategies=["left", "two-sided"],
        bounds={"lower": np.zeros(2), "upper": np.ones(2)},
    )

    expected = [
        np.array([0.15, 0.45]),
        np.array([0.15, 0.5]),
        np.array([0.15, 0.55]),
        np.array([0.2, 0.45]),
        np.array([0.2, 0.5]),
        np.array([0.2, 0.55]),
    ]

    for calc, exp in zip(calculated, expected):
        assert_array_almost_equal(calc, exp)


def test_get_direct_search_sample_binding_bounds():
    calculated = _get_direct_search_sample(
        current_x=np.array([0.2, 0.5]),
        step_size=0.05,
        search_strategies=["left", "two-sided"],
        bounds={"lower": np.zeros(2), "upper": np.ones(2) * 0.51},
    )

    expected = [
        np.array([0.15, 0.45]),
        np.array([0.15, 0.5]),
        np.array([0.2, 0.45]),
        np.array([0.2, 0.5]),
    ]

    for calc, exp in zip(calculated, expected):
        assert_array_almost_equal(calc, exp)
