"""按障碍的选边承诺与目标对齐规划窗口：只打破近似平局，不放松任何安全检查。"""
import numpy as np
import pytest
from omni_drones.navigation.forest.guidance import Guidance, GuidanceTracker
from omni_drones.navigation.forest.local import LocalPlanner
from omni_drones.navigation.forest.observed_esdf import ObservedESDF
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


def evidence():
    x, y, z = np.meshgrid(np.arange(-2, 6.1, .2), np.arange(-3.4, 3.41, .2), np.arange(-1, 1.6, .2))
    return np.c_[x.ravel(), y.ravel(), z.ravel()]


def column(x, y, radius=.15):
    angle, z = np.meshgrid(np.linspace(0, 2*np.pi, 16, endpoint=False), np.arange(-.8, 1.21, .1))
    return np.c_[x+radius*np.cos(angle).ravel(), y+radius*np.sin(angle).ravel(), z.ravel()]


def wall(y, x0=-1., x1=5.):
    x, z = np.meshgrid(np.arange(x0, x1+1e-9, .1), np.arange(-.8, 1.21, .1))
    return np.c_[x.ravel(), np.full(x.size, y), z.ravel()]


def scene(left_wall=2.2, right_wall=-2.2):
    return np.vstack([column(1.6, 0.), wall(left_wall), wall(right_wall)])


def tracker(points=((0, 0, 0), (8, 0, 0))):
    # 3 m lookahead puts the target beyond the obstacle, as in the full-forest stall.
    result = GuidanceTracker(Guidance('row', 1, 'first', np.asarray(points)), 3.)
    result.update(np.zeros(3), 0., 0.)
    return result


def field(points, position):
    return ObservedESDF(points, evidence(), resolution=.2,
                        bounds=np.array([position-[5., 4., 2.], position+[5., 4., 2.]]), max_voxels=500000)


def plan(planner, guide, t, points, position=(0., 0., 0.), yaw=0., use_field=False):
    position = np.asarray(position, dtype=float)
    guide.update(position, yaw, t)
    distance = field(points, position) if use_field else None
    return planner.plan(points, evidence(), position, yaw, np.zeros(3), np.zeros(3), guide,
                        geometry_only=True, planning_reserve=.15, timestamp=t, distance_field=distance,
                        clearance_weight=.35 if use_field else 0.)


def max_lateral(route):
    route = np.asarray(route)
    return route[np.argmax(np.abs(route[:, 1])), 1]


def test_commitment_keeps_one_side_while_yaw_and_pose_jitter():
    points = scene()
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, side_commitment=True, grid_heading='target'), tracker()
    route, decision, safety = plan(planner, guide, 0., points)
    assert decision['side_commitment'] == 'NEW' and decision['side_sign'] in (-1, 1)
    first = decision['side_sign']
    rng = np.random.default_rng(3)
    for step in range(1, 25):
        position = [.02*step, rng.uniform(-.04, .04), 0.]
        route, decision, safety = plan(planner, guide, .1*step, points, position, rng.uniform(-.6, .9))
        assert decision['side_sign'] == first
        assert np.sign(max_lateral(route)) == first
        assert safety.with_clearance_reserve(.15).clear(route)


def test_target_aligned_window_does_not_rotate_with_measured_yaw():
    points = scene()
    routes = []
    for yaw in (-.6, 0., .9):
        planner, guide = LocalPlanner(DEFAULT_FLIGHT, grid_heading='target'), tracker()
        route, decision, _ = plan(planner, guide, 0., points, yaw=yaw)
        assert decision['grid_heading_rad'] == pytest.approx(0.)
        routes.append(route)
    for route in routes[1:]:
        np.testing.assert_allclose(route, routes[0])


@pytest.mark.parametrize('left, right, expected', [(1.9, -2.7, -1), (2.7, -1.9, 1)])
def test_new_commitment_prefers_the_side_with_larger_observed_clearance(left, right, expected):
    points = scene(left, right)
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, side_commitment=True, grid_heading='target'), tracker()
    route, decision, safety = plan(planner, guide, 0., points, use_field=True)
    assert decision['side_sign'] == expected
    assert np.sign(max_lateral(route)) == expected
    assert safety.with_clearance_reserve(.15).clear(route)


def test_blocked_committed_side_switches_instead_of_forcing_through():
    open_scene = scene(2.6, -2.6)
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, side_commitment=True, grid_heading='target'), tracker()
    _, decision, _ = plan(planner, guide, 0., open_scene)
    side = decision['side_sign']
    assert side in (-1, 1)
    blocked = np.vstack([open_scene, column(1.6, 1.6*side, .4)])
    route, decision, safety = plan(planner, guide, .1, blocked)
    assert decision['side_commitment'] == 'SWITCHED_BLOCKED'
    assert decision['side_sign'] == -side
    assert np.sign(max_lateral(route)) == -side
    assert safety.with_clearance_reserve(.15).clear(route)
    # Reopening the original side does not flip back while the commitment holds.
    _, decision, _ = plan(planner, guide, .2, open_scene)
    assert decision['side_sign'] == -side


def test_commitment_releases_after_passing_the_obstacle():
    points = scene(2.6, -2.6)
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, side_commitment=True, grid_heading='target'), tracker()
    _, decision, _ = plan(planner, guide, 0., points)
    side, reasons = decision['side_sign'], []
    for step, x in enumerate(np.arange(.4, 4.81, .4), start=1):
        y = 1.2*side if 1. < x < 2.4 else .4*side
        _, decision, _ = plan(planner, guide, .1*step, points, [x, y, 0.])
        reasons.append(decision['side_commitment'])
    assert 'PASSED' in reasons
    assert reasons[-1] in ('NONE', 'PASSED')


def test_commitment_ends_when_the_target_direction_reverses():
    points = scene(2.6, -2.6)
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, side_commitment=True), tracker()
    plan(planner, guide, 0., points)
    guide.set(Guidance('row', 2, 'back', np.array([[0., 0., 0.], [-8., 0., 0.]])), now=.1)
    _, decision, _ = plan(planner, guide, .1, points, yaw=np.pi)
    assert decision['side_commitment'] == 'DIRECTION_CHANGED'


def test_side_commitment_requires_a_finite_observation_time():
    planner = LocalPlanner(DEFAULT_FLIGHT, side_commitment=True)
    with pytest.raises(ValueError, match='timestamp'):
        planner.plan([], evidence(), np.zeros(3), 0., np.zeros(3), np.zeros(3), tracker(), geometry_only=True)


@pytest.mark.parametrize('position', [(.17, .18, .13), (-.19, -.15, -.12), (.05, -.19, .18)])
@pytest.mark.parametrize('points', [np.empty((0, 3)), scene(2.6, -2.6)], ids=['open', 'column'])
def test_lattice_snapped_route_starts_at_the_vehicle_and_never_steps_back(position, points):
    # Run 4e5b0c28: the snapped start-cell centre sat up to half a cell behind and
    # below the vehicle; the spline through that kink left observed space and the
    # drone stopped with no_motion for 25 s on every goal.
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, grid_heading='target'), tracker()
    position = np.asarray(position)
    route, decision, safety = plan(planner, guide, 0., points, position)
    route = np.asarray(route)
    assert len(route) >= 3 and not decision['reason']
    np.testing.assert_allclose(route[0], position)
    toward = guide.target[:2]-position[:2]
    assert (route[1, :2]-position[:2])@toward > 0
    # Heights stay body-anchored: the first step holds the measured altitude.
    assert route[1, 2] == pytest.approx(position[2])
    assert safety.with_clearance_reserve(.15).clear(route)



@pytest.mark.parametrize('rise', [.8, -.8])
def test_route_reaches_target_height_early_not_on_its_last_step(rise):
    # Run 3a2722fb: a flat route that climbed only at its end left a near-vertical
    # climb at the goal, inside the unscanned cone above the 52 deg LiDAR.
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, grid_heading='target'), tracker(((0, 0, 0), (8, 0, rise)))
    route, decision, safety = plan(planner, guide, 0., np.empty((0, 3)))
    route = np.asarray(route)
    level = np.flatnonzero(np.abs(route[:, 2]-guide.target[2]) <= planner.resolution/2+1e-9)
    assert len(level) and route[level[0], 0] <= 1.2
    assert safety.clear(route)
    flat = LocalPlanner(DEFAULT_FLIGHT, grid_heading='target', height_weight=0.)
    late, _, _ = plan(flat, tracker(((0, 0, 0), (8, 0, rise))), 0., np.empty((0, 3)))
    assert np.flatnonzero(np.abs(np.asarray(late)[:, 2]-guide.target[2]) <= .2+1e-9)[0] > level[0]


def test_route_height_follows_gentle_guidance_instead_of_cell_steps():
    # A 0.3 m rise used to become one 0.4 m A* cell step (run 401c7d8c).
    points = ((0, 0, 0), (2, 0, .15), (4, 0, .3), (8, 0, .3))
    planner = LocalPlanner(DEFAULT_FLIGHT, grid_heading='target', max_climb_slope=.15)
    route, decision, safety = plan(planner, tracker(points), 0., np.empty((0, 3)))
    route = np.asarray(route)
    run = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
    assert (np.abs(np.diff(route[:, 2])) <= .15*run+1e-9).all()
    assert route[-1, 2] > .1 and not np.isclose(route[1:, 2] % .4, 0.).all()
    assert safety.with_clearance_reserve(.15).clear(route)
    stepped, _, _ = plan(LocalPlanner(DEFAULT_FLIGHT, grid_heading='target'), tracker(points), 0., np.empty((0, 3)))
    assert np.abs(np.diff(np.asarray(stepped)[:, 2])).max() >= .4-1e-9
