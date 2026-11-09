"""Distance-field integration changes route preference without authorizing unknowns."""
import numpy as np

from omni_drones.navigation.forest.guidance import Guidance, GuidanceTracker
from omni_drones.navigation.forest.local import LocalMap, LocalPlanner
from omni_drones.navigation.forest.observed_esdf import ObservedESDF
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


def guide():
    result = GuidanceTracker(Guidance('row', 1, 'first', np.array([[0., 0., 0.], [8., 0., 0.]])))
    result.update(np.zeros(3), 0., 0.)
    return result


def test_distance_cost_selects_the_wider_of_two_observed_passages():
    x, y, z = np.meshgrid(np.arange(-2, 5.1, .2), np.arange(-3, 3.1, .2), np.arange(-1, 1.6, .2))
    evidence = np.c_[x.ravel(), y.ravel(), z.ravel()]
    y, z = np.meshgrid(np.arange(-.4, .41, .1), np.arange(-.6, .81, .1))
    barrier = np.c_[np.full(y.size, 1.2), y.ravel(), z.ravel()]
    x, z = np.meshgrid(np.arange(-.4, 3.3, .2), np.arange(-.6, .81, .2))
    # Both sides have the same route length; the lower side is bounded by a wall.
    wall = np.c_[x.ravel(), np.full(x.size, -2.6), z.ravel()]
    points = np.vstack([barrier, wall])
    field = ObservedESDF(points, evidence)
    planner = LocalPlanner(DEFAULT_FLIGHT)
    baseline, _, safety = planner.plan(points, evidence, np.zeros(3), 0., np.zeros(3), np.zeros(3),
                                      guide(), geometry_only=True)
    route, decision, _ = planner.plan(points, evidence, np.zeros(3), 0., np.zeros(3), np.zeros(3),
        guide(), geometry_only=True, distance_field=field, clearance_weight=.35)
    assert baseline[:, 1].min() < -1.5
    assert route[:, 1].max() > 1.5
    assert decision['esdf_used'] and decision['esdf_astar_penalty_max'] > 0
    assert field.query(route).distance.min() > field.query(baseline).distance.min()+.1
    assert safety.clear(route) and safety.known(route, np.zeros(3))
    lengths = [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in (baseline, route)]
    np.testing.assert_allclose(*lengths)


def test_large_distance_cannot_create_a_route_without_ray_evidence():
    field = ObservedESDF([], [], bounds=[[-5., -4., -2.], [5., 4., 2.]])
    route, decision, _ = LocalPlanner(DEFAULT_FLIGHT).plan([], np.empty((0, 3)), np.zeros(3), 0., np.zeros(3),
        np.zeros(3), guide(), geometry_only=True, distance_field=field, clearance_weight=.35)
    assert decision['esdf_used'] and decision['reason'] == 'UNOBSERVED'
    assert len(route) == 1
    np.testing.assert_array_equal(decision['raw_velocity'], np.zeros(3))


def test_rebuilt_field_drops_expired_and_evicted_obstacles_instead_of_reusing_distances():
    memory = LocalMap([[0., 0., 0.]], seconds=1., rolling=True, window_size=(6., 6., 4.))
    memory.add(dict(timestamp=0., sensor_quaternion=np.array([1., 0., 0., 0.]),
        sensor_position=np.zeros(3), points_sensor=np.array([[1., 0., 0.]]), valid=np.array([True]),
        directions_world=np.array([[1., 0., 0.]]), distances=np.array([1.]), max_range=15.))
    old = ObservedESDF(*memory.snapshot(0., np.zeros(3)), bounds=[[-2., -2., -1.], [2., 2., 1.]])
    assert old.query([[1., 0., 0.]]).distance[0] < 0
    expired = ObservedESDF(*memory.snapshot(1.1, np.zeros(3)), bounds=[[-2., -2., -1.], [2., 2., 1.]])
    assert not expired.occupied.any() and not expired.known.any()
    memory.recenter([80., 0., 0.]); memory.recenter([0., 0., 0.])
    revisited = ObservedESDF(*memory.snapshot(.5, np.zeros(3)), bounds=[[-2., -2., -1.], [2., 2., 1.]])
    assert not revisited.occupied.any() and not revisited.known.any()
    # The active runtime snapshot is immutable after rolling memory changes.
    assert old.query([[1., 0., 0.]]).distance[0] < 0
