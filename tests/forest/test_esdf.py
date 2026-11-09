"""三维穿越、细枝覆盖和斜向碰撞的回归测试。"""
import numpy as np
import pytest
import trimesh
from omni_drones.navigation.forest.esdf import ESDF, astar, rasterize, gate_crossings
from omni_drones.navigation.forest.planning import smooth_path
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


def test_thin_triangle_between_centers():
    occupied = np.zeros((5, 5, 5), dtype=bool)
    triangle = np.array([[[1.49, .8, .8], [1.49, 3.2, .8], [1.49, .8, 3.2]]])
    rasterize(triangle, np.zeros(3), 1., occupied)
    assert occupied[1, 1, 1]
    assert not occupied[3].any()


def test_no_diagonal_corner_cutting():
    free = np.zeros((2, 2, 2), dtype=bool)
    free[0, 0, 0] = free[1, 1, 1] = True
    with pytest.raises(ValueError, match='no connected'):
        astar(free, np.zeros(free.shape), np.array([0, 0, 0]), np.array([1, 1, 1]))


def test_blocked_corridor_does_not_escape_to_adjacent_row():
    wall = trimesh.creation.box(extents=[.4, 4.2, 6.])
    wall.apply_translation([0., 0., 3.])
    field = ESDF(wall.vertices, wall.faces, lambda p: np.zeros(len(p)), [-3., 3., -2., 2.])
    with pytest.raises(ValueError, match='no connected'):
        field.route(np.array([-2., 0., 2.]), np.array([2., 0., 2.]), [])


def test_route_climbs_over_wall_and_preserves_clearance():
    wall = trimesh.creation.box(extents=[.4, 8., 2.2])
    wall.apply_translation([0., 0., 1.1])
    field = ESDF(wall.vertices, wall.faces, lambda p: np.zeros(len(p)), [-3., 3., -2., 2.])
    route = field.route(np.array([-2., 0., 2.]), np.array([2., 0., 2.]), [])
    assert route[:, 2].max() > 3.
    samples = np.concatenate([np.linspace(a, b, 101) for a, b in zip(route[:-1], route[1:])])
    assert field.clear(samples)
    assert wall.nearest.on_surface(samples)[1].min() > .75
    assert field.query(np.array([[0., 0., 1.]]))[0] < 0
    curve = smooth_path(route, DEFAULT_FLIGHT, field.clear)
    samples = curve.remaining(0., route[0])
    assert field.clear(samples)
    assert wall.nearest.on_surface(samples)[1].min() > .75


@pytest.mark.parametrize('side', [-1, 1])
def test_bypass_stays_in_corridor_and_returns_to_center(side):
    tree = trimesh.creation.box(extents=[.6, .6, 6.])
    tree.apply_translation([0., 0., 3.])
    field = ESDF(tree.vertices, tree.faces, lambda p: np.zeros(len(p)), [-4., 4., -2., 2.])
    gates = [dict(x=-2., y_min=-.4, y_max=.4), dict(x=0., y_min=min(side,2*side), y_max=max(side,2*side)),
             dict(x=2., y_min=-.4, y_max=.4)]
    free=field.free.copy()
    route = field.route(np.array([-3., 0., 2.]), np.array([3., 0., 2.]), gates)
    np.testing.assert_array_equal(field.free,free)
    assert gate_crossings(route, gates) is not None
    assert np.abs(route[:, 1]).max() <= 2.
    # 一直沿左侧行进虽然满足左绕，却没有按要求回中。
    assert gate_crossings(np.array([[-3., side*1.5, 2.], [3., side*1.5, 2.]]), gates) is None
    def safe(p):
        return field.clear(p) and gate_crossings(p, gates, require_all=False) is not None
    curve = smooth_path(route, DEFAULT_FLIGHT, safe)
    samples = curve.remaining(0., route[0])
    assert safe(samples) and gate_crossings(samples, gates) is not None
    assert tree.nearest.on_surface(samples)[1].min() > .75
