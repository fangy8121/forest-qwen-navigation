import json
import numpy as np
import pytest
from omni_drones.navigation.forest.geometry import parse_path, local_to_world
from omni_drones.navigation.forest.flight import local_goal
from omni_drones.navigation.forest.planning import PathSafety, RollingPlanner, smooth_path
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT
from omni_drones.navigation.forest.trajectory import Heading, QuinticPath


def evidence_volume():
    x, y, z = np.meshgrid(np.arange(-2, 10.1, .2), np.arange(-6, 6.1, .2), np.arange(-2, 2.1, .2))
    return np.c_[x.ravel(), y.ravel(), z.ravel()]


def test_vertical_goal_and_strict_xyz_protocol():
    np.testing.assert_allclose(local_goal([0, 0, 10], [0, 0, 2], 0), [0, 0, 1.3])
    goal = np.array([10., 2., 3.])
    local = local_goal(goal, np.zeros(3), 0)
    np.testing.assert_allclose(np.cross(local, goal), 0, atol=1e-12)
    for points in ([[1, 0]]*6, [[1, 0, 1.6]]*6, [[1, 0, True]]*6):
        with pytest.raises(ValueError):
            parse_path(json.dumps(dict(status='GO', points=points)))
    with pytest.raises(ValueError):
        parse_path('{"status":"STOP","points":[]}')


def test_cross_layer_segment_hits_branch_between_clear_endpoints():
    safety = PathSafety([[1., 0., .5]], evidence_volume(), None, DEFAULT_FLIGHT)
    path = np.array([[1., 0., -.5], [1., 0., 1.5]])
    assert safety.clear_points(path).all()
    assert not safety.clear(path)
    assert safety.clear(np.array([[0., 0., 0.], [2., 0., 0.]]))
    assert not safety.clear(np.array([[0., 0., .5], [2., 0., .5]]))


def test_xy_ray_plane_does_not_certify_climb_volume():
    evidence = evidence_volume()
    plane = evidence[np.abs(evidence[:, 2]) < .01]
    safety = PathSafety(np.empty((0, 3)), plane, None, DEFAULT_FLIGHT)
    assert not safety.known(np.array([[0., 0., 0.], [.5, 0., .5]]), np.zeros(3))
    safety = PathSafety(np.empty((0, 3)), evidence, None, DEFAULT_FLIGHT)
    assert safety.known(np.array([[0., 0., 0.], [.5, 0., .5]]), np.zeros(3))


def test_world_yaw_transform_retains_vertical_safety():
    position, yaw = np.array([10., -3., 4.]), 1.2
    points = np.array([[1., 0., .5]])
    runner = object.__new__(RollingPlanner)
    runner.flight, runner.lidar_hz = vars(DEFAULT_FLIGHT), 10.
    runner.observe(dict(points=points, evidence=evidence_volume(), position=position,
                        yaw=yaw, timestamp=0.))
    for z, expected in [(0., True), (.5, False)]:
        world = local_to_world([[0., 0., z], [2., 0., z]], position, yaw)
        assert runner.clear(world) == expected


def test_vertical_curve_and_braking_keep_xyz_and_heading():
    safety = PathSafety(np.empty((0, 3)), evidence_volume(), None, DEFAULT_FLIGHT)
    curve = smooth_path(np.array([[0., 0., 0.], [0., 0., 1.]]), DEFAULT_FLIGHT, safety.clear)
    heading = Heading(.7)
    assert heading.target(curve, curve.duration/2) == pytest.approx(.7)
    p, v, a = curve.sample(curve.duration/2)
    assert v[2] > 0 and np.linalg.norm(v[:2]) == 0
    brake = QuinticPath.brake(p, v, a)
    np.testing.assert_allclose(brake.sample(0)[1], v)
    assert brake.points[-1, 2] > p[2]
    np.testing.assert_allclose(brake.sample(brake.duration)[1], 0, atol=1e-10)
    assert safety.guard(curve, 0., np.zeros(3))


def test_observation_descent_requires_new_free_space_and_braking_coverage():
    safety = PathSafety(np.empty((0, 3)), evidence_volume(), None, DEFAULT_FLIGHT)
    curve = safety.observation_descent(np.zeros(3), np.zeros(3), .2)
    assert curve is not None
    np.testing.assert_allclose(curve.sample(0)[0], [0., 0., 0.])
    np.testing.assert_allclose(curve.sample(curve.duration)[0], [0., 0., -.2])
    np.testing.assert_allclose(curve.sample(curve.duration)[1], 0., atol=1e-10)
    for t in np.linspace(0., curve.duration, 25):
        assert safety.guard(curve, t, np.zeros(3), body_padding=0.)
    # 上方的射线不能借用当前机体外的 5 cm 余量，逐步探索下方未知区。
    evidence = evidence_volume()
    upper = evidence[np.abs(evidence[:, 2]-.2) < .01]
    safety = PathSafety(np.empty((0, 3)), upper, None, DEFAULT_FLIGHT)
    short = np.array([[0., 0., 0.], [0., 0., -.05]])
    assert safety.known(short, np.zeros(3))
    assert safety.observation_descent(np.zeros(3), np.zeros(3), .05) is None
    safety = PathSafety([[0., 0., -.5]], evidence, None, DEFAULT_FLIGHT)
    assert safety.observation_descent(np.zeros(3), np.zeros(3), .2) is None
