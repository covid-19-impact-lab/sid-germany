import numpy as np
from numpy.testing import assert_array_almost_equal

from src.manfred.linesearch import _find_maximal_linesearch_step
from src.manfred.linesearch import _normalize_direction


def test_find_maximal_line_search_step_negative_direction():
    x = np.array([0.5, 0.7, 0.2])
    direction = np.array([1, 1, -2])
    bounds = {"lower": np.zeros(3), "upper": np.ones(3)}
    max_step_size = 0.5
    calculated = _find_maximal_linesearch_step(x, direction, bounds, max_step_size)
    assert np.allclose(calculated, 0.1)


def test_find_maximal_line_search_step_binding_max_step():
    x = np.ones(3) * 0.5
    direction = np.ones(3) * 0.5
    bounds = {"lower": np.zeros(3), "upper": np.ones(3)}
    max_step_size = 0.1
    calculated = _find_maximal_linesearch_step(x, direction, bounds, max_step_size)
    assert np.allclose(calculated, 0.2)


def test_find_maximal_line_search_step_positive_direction():
    x = np.array([0.5, 0.7, 0.2])
    direction = np.array([1, 1, 1])
    bounds = {"lower": np.zeros(3), "upper": np.ones(3)}
    max_step_size = 0.5
    calculated = _find_maximal_linesearch_step(x, direction, bounds, max_step_size)
    assert np.allclose(calculated, 0.3)


def test_normalize_direction():
    direction = np.array([0, 1, 0])
    calculated = _normalize_direction(direction)
    assert_array_almost_equal(calculated, direction)
