import numpy as np
import pytest
from scipy.spatial import cKDTree
from omni_drones.navigation.forest.observed_esdf import ObservedESDF


def test_unknown_is_independent_of_empty_obstacle_distance_and_bounds(tmp_path):
    field = ObservedESDF([], [[0, 0, 0]], bounds=[[-1, -1, -1], [2, 1, 1]])
    query = field.query([[0, 0, 0], [1, 0, 0], [100, 0, 0]])
    assert query.distance[0] == query.distance[1] > 1
    np.testing.assert_array_equal(query.known, [True, False, False])
    np.testing.assert_array_equal(query.in_bounds, [True, True, False])
    assert query.distance[2] == 0
    field.save(tmp_path/'field.npz')
    with np.load(tmp_path/'field.npz') as archive:
        assert not archive['occupied'].any()
        assert archive['known'].sum() == 1


def test_occupied_evidence_conflict_is_unknown_and_near_obstacle_is_negative():
    field = ObservedESDF([[0, 0, 0]], [[0, 0, 0], [.8, 0, 0]])
    query = field.query([[0, 0, 0], [.8, 0, 0]])
    np.testing.assert_array_equal(query.known, [False, True])
    assert query.distance[0] < 0 < query.distance[1]


def test_trilinear_gradient_and_conservative_distance_to_unquantized_hits():
    rng = np.random.default_rng(91)
    hits = rng.uniform(-1, 1, (25, 3))
    field = ObservedESDF(hits, [], bounds=[[-2, -2, -2], [2, 2, 2]])
    points = rng.uniform(-1.9, 1.9, (100, 3))
    query = field.query(points)
    exact = cKDTree(hits).query(points)[0]
    assert (query.distance <= exact+1e-10).all()
    eps = 1e-6
    numeric = np.column_stack([(field.query(points+eps*axis).distance
                                - field.query(points-eps*axis).distance)/(2*eps)
                               for axis in np.eye(3)])
    np.testing.assert_allclose(query.gradient, numeric, atol=1e-7)
    assert not query.known.any()


def test_rejects_ambiguous_empty_map_nonfinite_input_and_oversize_grid():
    with pytest.raises(ValueError):
        ObservedESDF([], [])
    with pytest.raises(ValueError):
        ObservedESDF([[np.nan, 0, 0]], [])
    with pytest.raises(ValueError):
        ObservedESDF([], [], bounds=[[0, 0, 0], [1e5, 1e5, 1e5]])
