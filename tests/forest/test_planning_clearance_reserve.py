"""Extra planning room leaves the independent physical safety contract intact."""
from types import SimpleNamespace

import numpy as np
import pytest

from omni_drones.navigation.forest.guidance import Guidance, GuidanceTracker
from omni_drones.navigation.forest.local import LocalPlanner, VelocityTrajectory
from omni_drones.navigation.forest.observed_esdf import ObservedESDF
from omni_drones.navigation.forest.planning import PathSafety
from omni_drones.navigation.forest.spline_optimizer import PythonSplineOptimizer
from omni_drones.navigation.forest.spline_runtime import execution_check
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


def observations():
    grid = np.meshgrid(np.arange(-2., 5.1, .2), np.arange(-3., 3.1, .2),
                       np.arange(-1., 1.6, .2), indexing='ij')
    return np.stack(grid, axis=-1).reshape(-1, 3)


def test_planning_view_shares_evidence_without_changing_physical_clearance_or_frame():
    position = np.array([25., -3., 4.])
    points = position+np.array([[1., .8, 0.]])
    physical = PathSafety(points, observations()+position, None, DEFAULT_FLIGHT,
                          height_bounds=[3., 5.])
    physical.frame_position, physical.frame_yaw = position, np.pi/2
    reserved = physical.with_clearance_reserve(.15)
    target = position+np.array([[1., 0., 0.]])
    assert physical.clear(target)
    assert not reserved.clear(target)
    assert physical.clearance_reserve == 0.
    assert reserved.clearance_reserve == pytest.approx(.15)
    assert reserved.obstacles is physical.obstacles
    assert reserved.evidence is physical.evidence
    np.testing.assert_array_equal(reserved.frame_position, position)
    assert reserved.frame_yaw == physical.frame_yaw
    assert reserved.height_bounds is physical.height_bounds
    np.testing.assert_array_equal(physical.known_points(target, position),
                                  reserved.known_points(target, position))
    # A reserve can only remove clearance, including above/below a branch.
    vertical = position+np.array([[1., .8, .5]])
    assert physical.clear_points(vertical).all()
    assert not reserved.clear_points(vertical).all()
    assert reserved.with_clearance_reserve(.05).clearance_reserve == pytest.approx(.2)
    assert reserved.clearance_reserve == pytest.approx(.15)
    assert not reserved.clear_points(position+[[0., 0., 1.1]]).any()


@pytest.mark.parametrize('reserve', [-.01, np.nan, np.inf])
def test_invalid_reserve_does_not_create_a_planning_view(reserve):
    safety = PathSafety([], observations(), None, DEFAULT_FLIGHT)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        safety.with_clearance_reserve(reserve)
    assert safety.clearance_reserve == 0.


def test_astar_gives_a_wall_more_room_and_returns_the_original_physical_safety():
    hits = np.c_[np.arange(1., 3.01, .2), np.full(11, .9), np.zeros(11)]
    results = []
    for reserve in (0., .15):
        guide = GuidanceTracker(Guidance('row', 1, 'first',
                                          np.array([[0., 0., 0.], [8., 0., 0.]])))
        guide.update(np.zeros(3), 0., 0.)
        route, decision, safety = LocalPlanner(DEFAULT_FLIGHT, connect_start=True).plan(
            hits, observations(), np.zeros(3), 0., np.zeros(3), np.zeros(3), guide,
            geometry_only=True, planning_reserve=reserve)
        assert not decision['reason']
        assert decision['planning_clearance_reserve_m'] == reserve
        assert safety.clearance_reserve == 0.
        assert safety.with_clearance_reserve(reserve).clear(route)
        results.append(route)
    assert abs(results[0][:, 1]).max() < 1e-8
    assert results[1][:, 1].min() < -.3
    assert results[1][-1, 0] == pytest.approx(results[0][-1, 0])


def test_optimizer_cannot_accept_a_fixed_endpoint_that_consumes_planning_reserve():
    hits, evidence = np.array([[1.4, .8, 0.]]), observations()
    physical = PathSafety(hits, evidence, None, DEFAULT_FLIGHT)
    field = ObservedESDF(hits, evidence)
    optimizer = PythonSplineOptimizer(DEFAULT_FLIGHT)
    route = [[0., 0., 0.], [.8, 0., 0.], [1.4, 0., 0.]]
    base = optimizer.optimize(route, np.zeros(3), np.zeros(3), np.zeros(3), field, physical)
    reserved = optimizer.optimize(route, np.zeros(3), np.zeros(3), np.zeros(3),
                                  field, physical.with_clearance_reserve(.15))
    assert base.validation.accepted
    assert not reserved.validation.accepted
    assert reserved.validation.reason == 'OBSTACLE_OR_PUBLIC_BOUNDS'
    assert physical.clear(base.path.sample([0., base.path.duration])[0])


def test_actual_brake_warns_before_the_independent_physical_threshold_is_crossed():
    # A permitted lateral velocity tracking error sends the measured-state
    # brake closer to a branch than the still-clear nominal reference brake.
    # This demonstrates an engineering buffer, not a bounded tracking proof.
    physical = PathSafety([[.15, .86, 0.]], observations(), None, DEFAULT_FLIGHT)
    command = VelocityTrajectory(np.zeros(3), [.25, 0., 0.], np.zeros(3),
                                 [.25, 0., 0.], DEFAULT_FLIGHT)
    checked = execution_check(command, 0., physical, actual_position=np.zeros(3),
        actual_velocity=[.25, .25, 0.], actual_acceleration=np.zeros(3), horizon=.2,
        position_tolerance=.2, velocity_tolerance=.3)
    assert checked.accepted
    assert physical.with_clearance_reserve(.15).clear(checked.nominal_envelope)
    assert physical.clear(checked.actual_brake_envelope)
    assert not physical.with_clearance_reserve(.075).clear(checked.actual_brake_envelope)
    assert physical.clearance_reserve == 0.


def test_clearance_margin_matches_the_collision_check_at_every_padding():
    from omni_drones.navigation.forest.geometry import clearance_margin, collision_free
    rng = np.random.default_rng(0)
    hits = rng.uniform(-1.5, 1.5, (60, 3))
    centers = rng.uniform(-1.5, 1.5, (300, 3))
    margin = clearance_margin(hits, centers)
    for padding in (0., .05, .15, .3):
        np.testing.assert_array_equal(collision_free(hits, centers, padding), margin > padding)


# A stopped body just inside 0.80 m + padding from a stem (runs ed81189a,
# aa7fffd3: 0.86 m): it may leave along any curve that only gets farther.
STEM = np.array([[.84, 0., 0.]])


def leaving_safety():
    safety = PathSafety(STEM, observations(), None, DEFAULT_FLIGHT).with_clearance_reserve(.15)
    assert not safety.clear_points(np.zeros((1, 3)), padding=.05).all()
    return safety


@pytest.mark.parametrize('end, accepted', [
    ([-1.5, 0., 0.], True),     # straight away from the stem
    ([-1., -1., 0.], True),     # diagonal away
    ([0., 1.5, 0.], True),      # sideways: the stem only recedes
    ([.4, 0., 0.], False),      # toward the stem
    ([-.02, 0., 0.], False),    # ends inside the padding
])
def test_a_curve_may_start_inside_the_reserve_only_to_leave_it(end, accepted):
    safety = leaving_safety()
    curve = np.linspace(np.zeros(3), end, 40)
    assert safety.clear_leaving(curve, .07, np.zeros(3)) is accepted
    assert not safety.clear_points(curve, padding=.07).all()


def test_an_escape_cannot_come_back_closer_or_enter_the_body_envelope():
    safety = leaving_safety()
    back = np.r_[np.linspace([0., 0., 0.], [-.1, 0., 0.], 5), np.linspace([-.1, 0., 0.], [.05, 0., 0.], 5),
                 np.linspace([.05, 0., 0.], [-1.5, 0., 0.], 30)]
    assert not safety.clear_leaving(back, .07, np.zeros(3))
    inside = PathSafety([[.6, 0., 0.]], observations(), None, DEFAULT_FLIGHT).with_clearance_reserve(.15)
    assert not inside.clear_leaving(np.linspace([0., 0., 0.], [-1.5, 0., 0.], 40), .07, np.zeros(3))
    # Without a start the leading samples get no exemption.
    assert not safety.clear(np.linspace([0., 0., 0.], [-1.5, 0., 0.], 40))
    assert safety.clear(np.linspace([0., 0., 0.], [-1.5, 0., 0.], 40), start=np.zeros(3))


def test_optimizer_leaves_the_reserve_band_from_a_stop_beside_a_stem():
    from omni_drones.navigation.forest.geometry import clearance_margin
    safety = leaving_safety()
    field = ObservedESDF(STEM, observations())
    optimizer = PythonSplineOptimizer(DEFAULT_FLIGHT)
    for route in ([[0., 0., 0.], [-.8, 0., 0.], [-1.6, 0., 0.]],
                  [[0., 0., 0.], [.3, .5, 0.], [.9, 1.3, 0.]]):
        result = optimizer.optimize(route, np.zeros(3), np.zeros(3), np.zeros(3), field, safety)
        assert result.validation.accepted
        # Accepted only because the curve bends away before it turns.
        margin = clearance_margin(safety.obstacles, result.path.sample(np.linspace(0., result.path.duration, 200))[0])
        inside = np.flatnonzero(margin <= .2)
        assert (np.diff(margin[:inside.max()+2]) >= -2e-3).all()


@pytest.mark.parametrize('profile', [None, 'reference_sphere_v1'])
def test_batched_segment_checks_match_one_check_per_segment(profile):
    # Run 8ba92099: ~50 one-by-one checks took 0.22 s per observation.
    rng = np.random.default_rng(3)
    position = np.array([25., -3., 4.])
    points = position+np.array([[1.05, 0., 0.], [0., -1.1, .05], [-1.05, .3, -.1]])
    flight = SimpleNamespace(**dict(vars(DEFAULT_FLIGHT), safety_profile=profile))
    safety = PathSafety(points, observations()+position, None, flight).with_clearance_reserve(.15)
    safety.frame_position = position
    ends = position+rng.uniform([-.4, -.4, -.15], [.4, .4, .15], (200, 3))
    ends[0] = position
    expected = np.array([safety.clear([position, end]) for end in ends])
    assert 0 < expected.sum() < len(ends)
    np.testing.assert_array_equal(safety.clear_segments(position, ends), expected)
    assert safety.clear_segments(position, np.empty((0, 3))).shape == (0,)


@pytest.mark.parametrize('profile', [None, 'reference_sphere_v1'])
def test_batched_escape_checks_match_one_check_per_segment(profile):
    # Run 393cb1bd: parked inside the padding, every short step failed at the
    # vehicle's own position; only steps that leave the stem may pass.
    from omni_drones.navigation.forest.geometry import INFLATION, REFERENCE_CLEARANCE
    rng = np.random.default_rng(4)
    flight = SimpleNamespace(**dict(vars(DEFAULT_FLIGHT), safety_profile=profile))
    stem = [[(INFLATION if profile is None else REFERENCE_CLEARANCE)+.16, 0., 0.]]
    safety = PathSafety(stem, observations(), None, flight).with_clearance_reserve(.15)
    origin = np.zeros(3)
    assert safety.clear_points(origin[None]).all() and not safety.clear([origin, [-.3, 0., 0.]])
    ends = rng.uniform([-.4, -.4, -.15], [.4, .4, .15], (200, 3))
    expected = np.array([safety.clear([origin, end], start=origin) for end in ends])
    assert 0 < expected.sum() < len(ends)
    np.testing.assert_array_equal(safety.clear_segments(origin, ends, leaving=True), expected)
    # The plain check passes only steps short enough for their sample padding.
    plain = safety.clear_segments(origin, ends)
    assert (plain <= expected).all() and plain.sum() < expected.sum()
    assert np.linalg.norm(ends[plain], axis=1).max(initial=0.) < .05


@pytest.mark.parametrize('profile', [None, 'reference_sphere_v1'])
def test_escape_check_measures_distances_only_as_far_as_its_comparisons_need(profile, monkeypatch):
    # Run 2f3009e0: stopped between two row stems, each spline validation spent
    # 0.65 s of 1.0 s measuring distances out to 2 m, every candidate was older
    # than 0.8 s once ready, and the vehicle never moved again.
    from omni_drones.navigation.forest.geometry import INFLATION, REFERENCE_CLEARANCE
    rng = np.random.default_rng(16)
    flight = SimpleNamespace(**dict(vars(DEFAULT_FLIGHT), safety_profile=profile))
    clearance = INFLATION if profile is None else REFERENCE_CLEARANCE
    angle, radius = rng.uniform(0., 2*np.pi, 8), rng.uniform(1.6, 3., 8)
    stems = np.vstack([[clearance+.16, 0., 0.], [-.3, -(clearance+.12), 0.],
                       np.c_[radius*np.cos(angle), radius*np.sin(angle), rng.uniform(-.3, .3, 8)]])
    safety = PathSafety(stems, observations(), None, flight).with_clearance_reserve(.15)
    origin = np.zeros(3)
    assert safety.clear_points(origin[None]).all() and not safety.clear([origin, [.3, 0., 0.]])
    import omni_drones.navigation.forest.planning as planning
    caps, measure = [], planning.clearance_margin
    monkeypatch.setattr(planning, 'clearance_margin', lambda *a: caps.append(a[3]) or measure(*a))
    outcomes = []
    for i in range(300):
        first = origin if i % 2 else rng.uniform([-.3, -.3, -.1], [.3, .3, .1])
        path = first+np.cumsum(np.vstack([np.zeros(3), rng.uniform([-.4, -.4, -.1], [.4, .4, .1], (3, 3))]), axis=0)
        samples = np.concatenate([np.linspace(a, b, 8) for a, b in zip(path[:-1], path[1:])])
        free = safety.clear_points(samples, .05)
        full = safety._leaves(samples, free, origin, .05, require_exit=False, cap=2.)
        caps.clear()
        assert safety.clear_leaving(samples, .05, origin, require_exit=False) == full
        assert max(caps, default=0.) <= .05+.15+.05+1e-9
        outcomes.append((full, free.all()))
    assert {(True, False), (False, False), (True, True)} <= set(outcomes)
