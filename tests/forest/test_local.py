"""引导连续性、因果射线地图、速度动力学与局部执行安全协议。"""
import numpy as np
import pytest
from omni_drones.navigation.forest.guidance import Guidance, GuidanceTracker
from omni_drones.navigation.forest.local import LocalMap, LocalPlanner, VelocityTrajectory, certificate
from omni_drones.navigation.forest.planning import PathSafety
from omni_drones.navigation.forest.geometry import occupied_centers
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


def evidence():
    x, y, z = np.meshgrid(np.arange(-2, 5.1, .2), np.arange(-3, 3.1, .2), np.arange(-1, 1.6, .2))
    return np.c_[x.ravel(), y.ravel(), z.ravel()]


def tracker(points=((0, 0, 0), (8, 0, 0))):
    result = GuidanceTracker(Guidance('row', 1, 'first', np.asarray(points)))
    result.update(np.zeros(3), 0., 0.)
    return result


def test_target_moves_with_actual_progress_and_current_yaw():
    guide = tracker()
    for t in (1., 10., 100.):
        np.testing.assert_allclose(guide.update([0, 0, 0], 0., t), [2, 0, 0])
    np.testing.assert_allclose(guide.update([1, 0, 0], np.pi/2, 101.), [0, -2, 0], atol=1e-12)
    guide.update([.5, .4, .2], 0., 102.)
    assert guide.progress == 1.
    feedback = guide.feedback(102., 'BLOCKED', 'UNOBSERVED', .01)
    assert feedback.no_progress_seconds == 1.
    assert feedback.height_error == .2


def test_geometry_search_does_not_fit_an_unused_velocity_transition():
    route, decision, safety = LocalPlanner(DEFAULT_FLIGHT).plan([], evidence(), np.zeros(3), 0.,
        np.zeros(3), [5., 0., 0.], tracker(), geometry_only=True)
    assert len(route) > 1
    assert safety.clear(route)
    assert safety.known(route, np.zeros(3))
    assert np.linalg.norm(route[-1]-[2., 0., 0.]) < .5


def test_short_step_search_finds_lateral_detour_after_three_blocked_height_variants():
    # A branch leaves the three closest endpoints point-clear, while their
    # complete segments fail the unchanged sampling margin. The fourth, a
    # lateral step, is both observed and segment-clear; coarse cells cannot
    # express this narrow observed opening.
    grid = np.meshgrid(np.arange(-.8, .81, .1), np.arange(-.7, .71, .1),
                       np.arange(-.3, .31, .1))
    seen = np.stack(grid, axis=-1).reshape(-1, 3)
    origin = np.zeros(3)
    planner = LocalPlanner(DEFAULT_FLIGHT, connect_start=True)
    route, decision, physical = planner.plan([[.79, .67, .15]], seen, origin, 0.,
        origin, origin, tracker(((0., 0., 0.), (3., .9, 0.))),
        geometry_only=True, planning_reserve=.15)
    reserved = physical.with_clearance_reserve(.15)
    for height in (0., -.1, .1):
        endpoint = np.array([.3, 0., height])
        assert reserved.clear_points(endpoint[None]).all()
        assert not reserved.clear([origin, endpoint])
    assert len(route) == 2 and not decision['reason']
    np.testing.assert_allclose(route[-1], [.3, -.1, 0.], atol=1e-12)
    assert reserved.clear(route) and reserved.known(route, origin)
    assert np.linalg.norm(route[-1]-[3., .9, 0.]) < np.linalg.norm([3., .9, 0.])
    assert physical.clearance_reserve == 0.


def test_a_vehicle_parked_inside_the_padding_beside_a_stem_can_still_leave():
    # Run 393cb1bd: 1-4 cm inside the segment padding, every step and start
    # connection failed at the vehicle's own position; local_recovery_budget.
    from omni_drones.navigation.forest.geometry import INFLATION
    origin, stem = np.zeros(3), [[0., -(INFLATION+.16), 0.]]
    planner = LocalPlanner(DEFAULT_FLIGHT, connect_start=True)
    route, decision, physical = planner.plan(stem, evidence(), origin, 0., origin, origin, tracker(),
                                             geometry_only=True, planning_reserve=.15)
    reserved = physical.with_clearance_reserve(.15)
    assert reserved.clear_points(origin[None]).all() and not reserved.clear([origin, [.4, 0., 0.]])
    assert not decision['reason'] and len(route) > 1
    assert reserved.clear(route, start=origin) and reserved.known(route, origin)
    assert route[-1, 0] > 1.


def persistent_plan(planner, guide, timestamp, position=(0., 0., 0.), yaw=0., points=(), seen=None, **kwargs):
    position = np.asarray(position)
    guide.update(position, yaw, timestamp)
    return planner.plan(points, evidence() if seen is None else seen, position, yaw,
        np.zeros(3), np.zeros(3), guide, geometry_only=True, planning_reserve=.15, timestamp=timestamp, **kwargs)


def test_persistent_world_route_survives_yaw_change_without_rebuilding_astar(monkeypatch):
    import omni_drones.navigation.forest.local as module
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, route_persistence=True), tracker()
    original, _, _ = persistent_plan(planner, guide, 0.)
    def unexpected_search(*args):
        raise AssertionError('a still-observed route should not rebuild the rotating grid')
    monkeypatch.setattr(module, 'astar', unexpected_search)
    route, decision, physical = persistent_plan(planner, guide, .1, [.1, .02, 0.], .4)
    assert decision['route_reused'] and not decision['reason']
    np.testing.assert_allclose(route[0], [.1, .02, 0.])
    np.testing.assert_array_equal(route[-1], original[-1])
    assert physical.with_clearance_reserve(.15).clear(route)


@pytest.mark.parametrize('changed', ['obstacle', 'unknown', 'goal', 'deviation'])
def test_route_persistence_never_overrides_new_evidence_or_goal_scope(changed):
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, route_persistence=True), tracker()
    persistent_plan(planner, guide, 0.)
    options = {}
    expected = dict(obstacle='OBSTACLE_OR_BOUNDS', unknown='UNOBSERVED',
        goal='GOAL_CHANGED', deviation='DEVIATION')[changed]
    if changed == 'obstacle':
        options['points'] = [[1., 0., 0.]]
    elif changed == 'unknown':
        options['seen'] = np.empty((0, 3))
    elif changed == 'goal':
        guide.set(Guidance('row', 2, 'next', np.array([[0., 0., 0.], [0., 8., 0.]])), now=.1)
    else:
        options['position'] = [0., .6, 0.]
    _, decision, _ = persistent_plan(planner, guide, .1, **options)
    assert not decision['route_reused']
    assert decision['persistence_reset_reason'] == expected


def test_persistence_no_progress_budget_uses_actual_arc_and_source_time():
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, route_persistence=True), tracker()
    persistent_plan(planner, guide, 0.)
    assert persistent_plan(planner, guide, 1., [.06, 0., 0.])[1]['route_reused']
    assert persistent_plan(planner, guide, 2.5, [.06, 0., 0.])[1]['route_reused']
    decision = persistent_plan(planner, guide, 3.01, [.06, 0., 0.])[1]
    assert not decision['route_reused'] and decision['persistence_reset_reason'] == 'NO_PROGRESS'


def test_gate_hold_transition_replans_instead_of_reusing_a_route_through_the_gate():
    planner, guide = LocalPlanner(DEFAULT_FLIGHT, route_persistence=True), tracker()
    persistent_plan(planner, guide, 0., gate_hold=False)
    decision = persistent_plan(planner, guide, .1, gate_hold=True)[1]
    assert not decision['route_reused']
    assert decision['persistence_reset_reason'] == 'GOAL_CHANGED'
    assert persistent_plan(planner, guide, .2, gate_hold=True)[1]['route_reused']
    assert not persistent_plan(planner, guide, .3, gate_hold=False)[1]['route_reused']


def test_short_observed_step_is_kept_until_its_endpoint_is_nearly_reached():
    grid = np.meshgrid(np.arange(-.8, .81, .1), np.arange(-.7, .71, .1), np.arange(-.3, .31, .1))
    seen = np.stack(grid, axis=-1).reshape(-1, 3)
    planner = LocalPlanner(DEFAULT_FLIGHT, connect_start=True, route_persistence=True)
    guide, hits = tracker(((0., 0., 0.), (3., .9, 0.))), [[.79, .67, .15]]
    original, _, _ = persistent_plan(planner, guide, 0., points=hits, seen=seen)
    assert len(original) == 2
    route, decision, _ = persistent_plan(planner, guide, .1, yaw=.1, points=hits, seen=seen)
    assert decision['route_reused']
    np.testing.assert_array_equal(route[-1], original[-1])
    near_end = .9*original[-1]
    decision = persistent_plan(planner, guide, .2, position=near_end, points=hits, seen=seen)[1]
    assert not decision['route_reused'] and decision['persistence_reset_reason'] == 'NEAR_END'


def aisle_evidence(x_max):
    x, y, z = np.meshgrid(np.arange(-2, x_max+.01, .2), np.arange(-3, 3.1, .2), np.arange(-1, 1.6, .2))
    return np.c_[x.ravel(), y.ravel(), z.ravel()]


@pytest.mark.parametrize('refresh', [None, 3.])
def test_lookahead_limited_route_is_replaced_before_its_stop_brakes(refresh):
    planner = LocalPlanner(DEFAULT_FLIGHT, route_persistence=True, grid_forward=4.4,
                           route_refresh_remaining=refresh)
    guide, seen = GuidanceTracker(Guidance('row', 1, 'first', np.array([[0., 0, 0], [12., 0, 0]])), 4.), aisle_evidence(8.)
    original = persistent_plan(planner, guide, 0., seen=seen)[0]
    assert np.linalg.norm(original[-1]-guide.target) < .6
    assert persistent_plan(planner, guide, .1, position=[.5, 0., 0.], seen=seen)[1]['route_reused']
    route, decision, _ = persistent_plan(planner, guide, .2, position=[1.2, 0., 0.], seen=seen)
    if refresh is None:
        assert decision['route_reused'] and np.array_equal(route[-1], original[-1])
    else:
        assert not decision['route_reused'] and decision['persistence_reset_reason'] == 'EXTEND'
        assert route[-1][0] > original[-1][0]+1.
        assert persistent_plan(planner, guide, .3, position=[1.4, 0., 0.], seen=seen)[1]['route_reused']


def test_route_cut_short_by_unknown_space_is_not_refreshed_every_scan():
    planner = LocalPlanner(DEFAULT_FLIGHT, route_persistence=True, grid_forward=4.4, route_refresh_remaining=3.)
    guide, seen = GuidanceTracker(Guidance('row', 1, 'first', np.array([[0., 0, 0], [12., 0, 0]])), 4.), aisle_evidence(2.6)
    original = persistent_plan(planner, guide, 0., seen=seen)[0]
    assert np.linalg.norm(original[-1]-guide.target) > 1.
    for timestamp, x in ((.1, .5), (.2, 1.2)):
        route, decision, _ = persistent_plan(planner, guide, timestamp, position=[x, 0., 0.], seen=seen)
        assert decision['route_reused'] and np.array_equal(route[-1], original[-1])


@pytest.mark.parametrize('value', [.5, -1., np.nan])
def test_route_refresh_must_lie_beyond_the_near_end_distance(value):
    with pytest.raises(ValueError, match='route_refresh_remaining'):
        LocalPlanner(DEFAULT_FLIGHT, route_refresh_remaining=value)


def test_enabled_persistence_requires_a_real_finite_observation_time():
    planner = LocalPlanner(DEFAULT_FLIGHT, route_persistence=True)
    for timestamp in (None, np.nan, np.inf):
        with pytest.raises(ValueError, match='timestamp'):
            planner.plan([], evidence(), np.zeros(3), 0., np.zeros(3), np.zeros(3), tracker(),
                         geometry_only=True, timestamp=timestamp)


def test_old_revision_is_ignored_and_new_segment_resets_stall():
    guide = tracker()
    assert not guide.set(Guidance('row', 1, 'stale', np.array([[0,0,0], [0,8,0]])))
    assert guide.set(Guidance('row', 2, 'second', np.array([[0,0,0], [0,8,0]])), now=20.)
    guide.update([0,0,0], 0., 21.)
    assert guide.feedback(21., 'RUNNING', '', .01).no_progress_seconds == 1.
    np.testing.assert_allclose(guide.target, [0,2,0])


@pytest.mark.parametrize('traversal', ['sampled_v1', 'voxel_dda_v1'])
def test_ray_stops_at_hit_and_memory_expires(traversal):
    memory = LocalMap([[0,0,0], [8,0,0]], seconds=1., ray_traversal=traversal)
    frame = dict(timestamp=0., sensor_quaternion=np.array([1.,0,0,0]),
                 sensor_position=np.zeros(3), points_sensor=np.array([[.6,0,0]]), valid=np.array([True]),
                 directions_world=np.array([[1.,0,0]]), distances=np.array([.6]), max_range=15.)
    memory.add(frame)
    hits, seen = memory.snapshot(0., np.zeros(3))
    np.testing.assert_allclose(hits, [[.6,0,0]])
    assert len(seen) and seen[:,0].max() < .6
    assert not len(memory.snapshot(1.01, np.zeros(3))[1])
    assert not len(memory.snapshot(1.01, np.zeros(3))[0])


def test_exact_traversal_recovers_briefly_crossed_voxels_without_seeing_behind_hit():
    route = [[0, 0, 0], [2, 0, 0]]
    direction = np.array([1., .37, .13]); direction /= np.linalg.norm(direction)
    origin = np.array([.09, .09, .09]); distance = 1.7
    frame = dict(timestamp=0., sensor_quaternion=np.array([1., 0., 0., 0.]),
                 sensor_position=origin, points_sensor=np.array([distance*direction]), valid=np.array([True]),
                 directions_world=direction[None], distances=np.array([distance]), max_range=15.)
    sampled = LocalMap(route)
    exact = LocalMap(route, ray_traversal='voxel_dda_v1')
    sampled.add(frame); exact.add(frame)
    # Dense independent sampling witnesses the short intersections missed at 15 cm spacing.
    samples = origin+np.arange(0., distance, .0001)[:, None]*direction
    witness = set(map(tuple, exact._cells(samples)))
    observed = set(map(tuple, np.argwhere(np.isfinite(exact.seen))))
    assert observed == witness
    assert len(observed-set(map(tuple, np.argwhere(np.isfinite(sampled.seen))))) > 0
    hit_cell = exact._cells(origin[None]+distance*direction)[0]
    evidence = exact.snapshot(0., origin)[1]
    assert tuple(hit_cell) not in set(map(tuple, exact._cells(evidence)))


@pytest.mark.parametrize('target', ([.8,0,0], [0,.6,.4], [0,0,0]))
def test_velocity_transition_and_braking_limits(target):
    path = VelocityTrajectory(np.zeros(3), np.array([.3,0,0]), np.array([.1,0,0]), target, DEFAULT_FLIGHT)
    times = np.linspace(0, path.duration, 501)
    p, v, a = path.sample(times)
    np.testing.assert_allclose(p[0], 0., atol=1e-12)
    np.testing.assert_allclose(v[0], [.3,0,0], atol=1e-12)
    np.testing.assert_allclose(a[0], [.1,0,0], atol=1e-12)
    np.testing.assert_allclose(v[-1], target, atol=1e-12)
    np.testing.assert_allclose(a[-1], 0., atol=1e-12)
    assert np.linalg.norm(a, axis=1).max() <= 1.001
    assert np.linalg.norm(path.curve(times, 2), axis=1).max() <= 4.001
    assert np.linalg.norm(v, axis=1).max() <= 1.001


def test_certificate_includes_full_brake_after_execution_prefix():
    path = VelocityTrajectory(np.zeros(3), np.array([.6,0,0]), np.zeros(3), [.6,0,0], DEFAULT_FLIGHT)
    clear = PathSafety(np.empty((0,3)), evidence(), None, DEFAULT_FLIGHT)
    assert certificate(clear, path, np.zeros(3))[0] == ''
    blocked = PathSafety([[.9,0,0]], evidence(), None, DEFAULT_FLIGHT)
    reason, envelope = certificate(blocked, path, np.zeros(3))
    assert envelope[:,0].max() > .2*.6
    assert reason == 'OBSTACLE'
    unknown = PathSafety(np.empty((0,3)), [[0,0,0]], None, DEFAULT_FLIGHT)
    assert certificate(unknown, path, np.zeros(3))[0] == 'UNOBSERVED'


def test_classic_planner_advances_in_clear_observed_space():
    planner = LocalPlanner(DEFAULT_FLIGHT)
    path, decision, _ = planner.plan(np.empty((0,3)), evidence(), np.zeros(3), 0.,
                                  np.zeros(3), np.zeros(3), tracker())
    assert decision['certified'] and decision['reason'] == ''
    assert decision['raw_velocity'][0] > .5
    assert path.sample(.2)[0][0] > 0


def test_classic_planner_routes_around_observed_barrier():
    planner = LocalPlanner(DEFAULT_FLIGHT)
    y, z = np.meshgrid(np.arange(-.4,.41,.1), np.arange(-.6,.81,.1))
    points = np.c_[np.full(y.size,1.2), y.ravel(), z.ravel()]
    _, decision, _ = planner.plan(points, evidence(), np.zeros(3), 0., np.zeros(3), np.zeros(3), tracker())
    assert decision['certified']
    assert abs(decision['raw_velocity'][1]) > .1
    assert decision['raw_velocity'][0] > 0


def test_unknown_space_produces_explicit_feedback():
    planner = LocalPlanner(DEFAULT_FLIGHT)
    _, decision, _ = planner.plan(np.empty((0,3)), [[0,0,0]], np.zeros(3), 0.,
                               np.zeros(3), np.zeros(3), tracker())
    assert decision['certified'] and decision['reason'] == 'UNOBSERVED'
    np.testing.assert_allclose(decision['executed_velocity'], 0.)


def test_fast_collision_query_matches_exact_cylinder_with_multiple_hits():
    rng = np.random.default_rng(45)
    points = rng.uniform(-2, 2, (3000, 3))
    centers = rng.uniform(-3, 3, (1000, 3))
    delta = points[None]-centers[:,None]
    expected = ((np.linalg.norm(delta[:,:,:2], axis=2) <= .7) & (np.abs(delta[:,:,2]) <= .45)).any(axis=1)
    np.testing.assert_array_equal(occupied_centers(points, centers, .7, .45), expected)
    assert not occupied_centers(points, np.empty((0,3))).any()


def test_short_observed_step_is_not_lost_to_coarse_grid():
    seen = evidence()
    seen = seen[seen[:,0] < .65]
    planner = LocalPlanner(DEFAULT_FLIGHT)
    _, decision, _ = planner.plan(np.empty((0,3)), seen, np.zeros(3), 0.,
                                  np.zeros(3), np.zeros(3), tracker())
    assert decision['certified'] and decision['reason'] == ''
    assert decision['raw_velocity'][0] > 0.


def test_mission_height_limit_is_checked_by_safety():
    safety = PathSafety(np.empty((0,3)), evidence(), None, DEFAULT_FLIGHT, height_bounds=(-.2,.2))
    assert safety.clear([[0,0,0], [1,0,0]])
    assert not safety.clear([[0,0,0], [1,0,.3]])


def test_body_padding_does_not_authorize_exploration():
    safety = PathSafety(np.empty((0,3)), [[0,0,0]], None, DEFAULT_FLIGHT)
    path = VelocityTrajectory(np.zeros(3), [.05,0,0], np.zeros(3), [.05,0,0], DEFAULT_FLIGHT)
    reason, envelope = certificate(safety, path, np.zeros(3))
    assert safety.known(envelope, np.zeros(3), body_padding=.05)
    assert reason == 'UNOBSERVED'


def test_overspeed_is_reported_as_dynamics():
    safety = PathSafety(np.empty((0,3)), evidence(), None, DEFAULT_FLIGHT)
    path = VelocityTrajectory(np.zeros(3), [1.1,0,0], np.zeros(3), [1.1,0,0], DEFAULT_FLIGHT)
    assert certificate(safety, path, np.zeros(3))[0] == 'DYNAMICS'


def test_actual_pose_can_connect_to_grid_without_clearing_an_occupied_cell():
    # Pose is > .65 m from the obstacle, but its coarse cell's extra padding fails.
    # A safe sideways connection still exists and must not require a forward step.
    points = np.array([[.8, 0., z] for z in np.arange(-.6, 1.4, .1)])
    planner = LocalPlanner(DEFAULT_FLIGHT, connect_start=True)
    path, decision, safety = planner.plan(points, evidence(), np.zeros(3), 0.,
                                         np.zeros(3), np.zeros(3), tracker())
    assert not safety.clear_points([[0., 0., 0.]], padding=.2)[0]
    assert safety.clear_points([[0., 0., 0.]])[0]
    assert decision['certified']
    assert np.linalg.norm(decision['executed_velocity']) > .05
    assert abs(decision['executed_velocity'][1]) > .05
    assert certificate(safety, path, np.zeros(3))[0] == ''


def test_virtual_start_does_not_authorize_unknown_connections():
    planner = LocalPlanner(DEFAULT_FLIGHT, connect_start=True)
    _, decision, _ = planner.plan([[.8, 0., 0.]], [[0., 0., 0.]], np.zeros(3), 0.,
                                  np.zeros(3), np.zeros(3), tracker())
    np.testing.assert_allclose(decision['executed_velocity'], 0.)
    assert decision['action'] == 'BRAKE'


def flat_ground(height):
    return lambda xy: (np.full(len(xy), float(height)), np.ones(len(xy), bool))


def low_bar(y_max=3., top=-.1):
    # A branch across the route just below the guidance height; at the default
    # top only the +0.8 m A* level clears it vertically.
    y, z = np.meshgrid(np.arange(-3., y_max+.01, .1), np.arange(top-.3, top+.05, .1))
    return np.c_[np.full(y.size, 1.6), y.ravel(), z.ravel()]


def test_routes_climb_over_a_low_branch_only_up_to_the_ground_ceiling():
    # Runs on test/000 and train/008: climbs over obstacles reached 3.55 m
    # above the ground against the 3.5 m task limit.
    free, _, _ = LocalPlanner(DEFAULT_FLIGHT, connect_start=True).plan(
        low_bar(), evidence(), np.zeros(3), 0., np.zeros(3), np.zeros(3), tracker(), geometry_only=True)
    assert free[:, 2].max() > .5
    planner = LocalPlanner(DEFAULT_FLIGHT, connect_start=True, agl_band=(2., 2.8))
    route, decision, _ = planner.plan(low_bar(), evidence(), np.zeros(3), 0., np.zeros(3), np.zeros(3),
                                      tracker(), geometry_only=True, ground=flat_ground(-2.5))
    assert route[:, 2].max() <= .3+1e-9 and route[:, 0].max() < 1.6
    assert decision['agl_band_removed_cells'] > 0


def test_a_ceiling_between_two_grid_levels_still_lets_the_route_over_a_branch():
    # Run 853c339a: clearing a branch needed 3.38 m against the 3.35 m ceiling,
    # the level below was too low, and the vehicle shuffled in place for 100 s.
    # Here 0.4 m does not clear the bar (0.6 m does), 0.8 m is above the 0.7 m ceiling.
    planner = LocalPlanner(DEFAULT_FLIGHT, connect_start=True, agl_band=(2., 3.2))
    route, decision, physical = planner.plan(low_bar(), evidence(), np.zeros(3), 0., np.zeros(3), np.zeros(3),
                                             tracker(), geometry_only=True, ground=flat_ground(-2.5))
    assert not decision['reason'] and route[-1, 0] > 1.6
    assert .6 <= route[:, 2].max() <= .7+1e-9
    assert physical.with_clearance_reserve(.15).clear(route, start=np.zeros(3))


def test_a_ground_ceiling_sends_the_route_around_a_branch_instead_of_over_it():
    bar = low_bar(y_max=1., top=-.3)
    over, _, _ = LocalPlanner(DEFAULT_FLIGHT, connect_start=True).plan(
        bar, evidence(), np.zeros(3), 0., np.zeros(3), np.zeros(3), tracker(), geometry_only=True)
    assert over[:, 2].max() > .3 and over[:, 1].max() < .5
    planner = LocalPlanner(DEFAULT_FLIGHT, connect_start=True, agl_band=(2., 2.6))
    route, decision, physical = planner.plan(bar, evidence(), np.zeros(3), 0., np.zeros(3), np.zeros(3),
                                             tracker(), geometry_only=True, ground=flat_ground(-2.5))
    assert not decision['reason']
    assert route[:, 2].max() <= .1+1e-9 and route[:, 1].max() > 1.5 and route[-1, 0] > 1.6
    assert physical.clear(route)


def test_the_guidance_and_the_vehicle_stay_allowed_outside_the_ground_band():
    # Slope-limited guidance can lag steep terrain beyond the band; it stays
    # flyable. A vehicle above the band is routed back down, not stranded.
    planner = LocalPlanner(DEFAULT_FLIGHT, connect_start=True, agl_band=(1.65, 2.8))
    route, decision, _ = planner.plan([], evidence(), np.zeros(3), 0., np.zeros(3), np.zeros(3), tracker(),
                                      geometry_only=True, ground=flat_ground(-3.))
    assert not decision['reason'] and route[-1, 0] > 1. and np.abs(route[:, 2]).max() < 1e-9
    high = np.array([0., 0., .6])
    planner = LocalPlanner(DEFAULT_FLIGHT, connect_start=True, agl_band=(1.65, 2.85))
    route, decision, _ = planner.plan([], evidence(), high, 0., np.zeros(3), np.zeros(3), tracker(),
                                      geometry_only=True, ground=flat_ground(-2.5))
    assert not decision['reason'] and route[-1, 0] > 1. and route[-1, 2] <= .35+1e-9


