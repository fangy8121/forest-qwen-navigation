"""Reference geometry, observation uncertainty and executable XYZ supervision."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from omni_drones.navigation.forest.geometry import REFERENCE_PROFILE, center_clearance
from omni_drones.navigation.forest.planning import PathSafety
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT
from omni_drones.navigation.forest.waypoint_teacher_xyz import centered_teacher, assess_answer, observed_grid
from omni_drones.navigation.forest.waypoint_teacher_xyz import reference_teacher, reference_prefix
from omni_drones.navigation.forest.trajectory import QuinticPath


def observation(goal=(6., 0., 0.), velocity=(0., 0., 0.), plane=False):
    x, y, z = np.meshgrid(np.arange(-2., 9., .2), np.arange(-3., 3.1, .2),
                          [0.] if plane else np.arange(-.8, 1.21, .2), indexing='ij')
    return dict(points=np.empty((0, 3)), evidence=np.c_[x.ravel(), y.ravel(), z.ravel()],
                goal=np.array(goal), velocity=np.array(velocity), acceleration=np.zeros(3),
                flight_json=json.dumps(dict(vars(DEFAULT_FLIGHT), safety_profile=REFERENCE_PROFILE)))


def test_reference_sphere_matches_distance_not_cylinder_axes():
    flight = SimpleNamespace(**dict(vars(DEFAULT_FLIGHT), safety_profile=REFERENCE_PROFILE))
    s = PathSafety([[0., 0., .6]], np.empty((0, 3)), None, flight)
    assert not s.clear_points([[0., 0., 0.]])[0]
    # A nearby corner is outside the sphere despite lying inside its bounding box.
    s = PathSafety([[.6, 0., .6]], np.empty((0, 3)), None, flight)
    assert s.clear_points([[0., 0., 0.]])[0]
    legacy = PathSafety([[0., 0., .6]], np.empty((0, 3)), None, DEFAULT_FLIGHT)
    assert legacy.clear_points([[0., 0., 0.]])[0]
    with pytest.raises(ValueError, match='unknown safety profile'):
        center_clearance({'safety_profile': 'typo'})


def test_missing_observation_is_review_not_a_stop_label():
    obs = observation()
    obs['evidence'] = np.empty((0, 3))
    answer, audit = centered_teacher(obs)
    assert answer['status'] == 'REVIEW' and not audit['accepted']
    assert audit['initial_braking']['obstacle_clear']
    assert not audit['initial_braking']['coverage_known']


def test_grid_does_not_add_voxel_box_radius_to_reference_clearance():
    obs = observation()
    obs['points'] = np.array([[0., .84, 0.]])
    grid, safety = observed_grid(obs)
    start = tuple(np.rint(-grid['origin']/grid['step']).astype(int))
    assert grid['search'][start]
    assert safety.clear(np.array([[0., 0., 0.], [1., 0., 0.]]))
    # A detour toward the obstacle must still fail the continuous path check.
    assert not safety.clear(np.array([[0., 0., 0.], [0., .15, 0.], [1., 0., 0.]]))


def test_reference_profile_agrees_in_bev_and_transformed_executor():
    from omni_drones.navigation.forest.structured_bev import extract_structured_bev
    from omni_drones.navigation.forest.planning import RollingPlanner
    from omni_drones.navigation.forest.geometry import local_to_world
    obs = observation()
    obs['points'] = np.array([[1., 0., .6]])
    image = extract_structured_bev(obs['points'], obs['evidence'], center_clearance=.75)
    cell = (np.argmin(abs(image.xs-1.)), np.argmin(abs(image.ys)))
    assert image.blocked[cell]
    position, yaw = np.array([10., -3., 4.]), 1.2
    runner = object.__new__(RollingPlanner)
    runner.flight, runner.lidar_hz = json.loads(obs['flight_json']), 10.
    runner.observe(dict(obs, position=position, yaw=yaw, timestamp=0.))
    world = local_to_world([[0., 0., 0.], [2., 0., 0.]], position, yaw)
    assert not runner.clear(world)


def test_goal_hold_requires_a_verified_stop_near_goal():
    answer, audit = centered_teacher(observation(goal=(.1, 0., 0.)))
    assert answer['status'] == 'HOLD' and audit['accepted']
    answer, audit = centered_teacher(observation(goal=(.1, 0., 0.), velocity=(1., 0., 0.)))
    assert answer['status'] == 'REVIEW' and audit['reason'] == 'goal_stop_unverified'


def test_descending_stop_template_does_not_prevent_level_recovery():
    obs = observation(velocity=(1., 0., -.1), plane=True)
    answer, audit = centered_teacher(obs)
    assert not audit['initial_braking']['coverage_known']
    assert answer['status'] == 'GO'
    assert assess_answer(answer, obs)['accepted']


def test_reference_is_teacher_only_and_reconstructed_before_acceptance():
    obs = observation(goal=(6., .5, .3))
    reference = np.linspace([0., 0., 0.], [6., .5, .3], 121)
    answer, audit = reference_teacher(obs, reference)
    assert answer['status'] == 'GO' and audit['source'] == 'reference'
    assert assess_answer(answer, obs)['accepted']
    np.testing.assert_allclose(answer['points'][-1], reference[-1]*6/np.linalg.norm(reference[-1]), atol=1e-4)
    assert 'reference' not in obs


def test_reference_clock_uses_settle_delay_and_actual_pose():
    curve = QuinticPath(np.array([[10., 2., 3.], [16., 2., 4.]]))
    prefix = reference_prefix(curve, 3., [10., 2., 3.], np.pi/2, settle_seconds=3., horizon=2.)
    np.testing.assert_allclose(prefix[0], 0.)
    assert abs(prefix[-1, 0]) < 1e-8 and prefix[-1, 1] < 0 and prefix[-1, 2] > 0
    assert np.linalg.norm(np.diff(prefix, axis=0), axis=1).sum() == pytest.approx(2.)


def test_voxel_traversal_matches_analytic_ray_box_intersections():
    from omni_drones.navigation.forest.ray_evidence import trace_free_voxels
    rng = np.random.default_rng(27)
    lower, step, shape = np.zeros(3), .2, (9, 8, 7)
    cells = np.indices(shape).reshape(3, -1).T
    lo, hi = cells*step-step/2, cells*step+step/2
    for _ in range(30):
        origin = rng.uniform(-1., 2., 3)
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        distance = rng.uniform(.1, 4.)
        seen = np.zeros(shape, bool)
        trace_free_voxels(seen, origin, d[None], np.array([distance]), lower, step)
        near, far = (lo-origin)/d, (hi-origin)/d
        entry = np.maximum(np.minimum(near, far).max(axis=1), 0.)
        exit = np.minimum(np.maximum(near, far).min(axis=1), distance)
        np.testing.assert_array_equal(seen.ravel(), exit-entry > 1e-9)


def test_voxel_rays_stop_at_hits_and_do_not_use_future_scans():
    from omni_drones.navigation.forest.observation import History
    h = History(center_clearance=.75, ray_traversal='voxel_dda_v1')
    frame = dict(timestamp=0., points_sensor=np.array([[1., 0., 0.]]), valid=np.array([True]),
                 sensor_position=np.zeros(3), sensor_quaternion=np.array([1., 0., 0., 0.]),
                 directions_world=np.array([[1., 0., 0.]]), distances=np.array([1.]), max_range=5.)
    h.add(frame)
    h.add(dict(frame, timestamp=1., points_sensor=np.array([[4., 0., 0.]]), distances=np.array([4.])))
    points, evidence = h.snapshot(0., np.zeros(3), 0.)
    np.testing.assert_allclose(points, [[1., 0., 0.]])
    assert evidence[:, 0].max() < 1.
