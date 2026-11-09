from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
from omni_drones.navigation.forest.catkin_spline import CatkinSpline
from omni_drones.navigation.forest.observed_esdf import ObservedESDF
from omni_drones.navigation.forest.planning import PathSafety
from omni_drones.navigation.forest.spline_optimizer import (
    endpoint_controls, validate_spline, PythonSplineOptimizer, stop_profile)
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


def observations():
    grid = np.meshgrid(np.arange(-1.6, 4.01, .2), np.arange(-2, 2.01, .2),
                       np.arange(-.8, .81, .2), indexing='ij')
    return np.stack(grid, axis=-1).reshape(-1, 3)


@pytest.mark.parametrize('interval', [.15, .6, 1.3])
def test_endpoint_state_is_exact_even_after_retiming(interval):
    p, v, a, end = np.array([[.1, -.2, 0], [.3, -.1, .05], [.1, .15, -.04], [2, .5, .3]])
    path = CatkinSpline(endpoint_controls(p, v, a, end, interval), interval)
    np.testing.assert_allclose(path.sample(0.), [p, v, a], atol=1e-12)
    np.testing.assert_allclose(path.sample(path.duration), [end, np.zeros(3), np.zeros(3)], atol=1e-12)


def test_validator_rejects_curve_collision_even_when_all_control_points_are_clear():
    points = np.array([[0, 0, 0]]*3+[[0, 2, 0], [2, 2, 0]]+[[2, 0, 0]]*3)
    path = CatkinSpline(points, 2.)
    hit = path.sample(path.duration/2)[0]
    flight = SimpleNamespace(max_speed=100., max_acceleration=100., max_jerk=100.)
    safety = PathSafety([hit], observations(), None, flight)
    assert safety.clear_points(points).all()
    result = validate_spline(path, safety, points[0])
    assert not result.accepted and result.reason == 'OBSTACLE_OR_PUBLIC_BOUNDS'


def test_empty_obstacles_do_not_authorize_a_curve_through_unknown_space():
    path = CatkinSpline(endpoint_controls([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], 1.), 1.)
    result = validate_spline(path, PathSafety([], [[0, 0, 0]], None, DEFAULT_FLIGHT), np.zeros(3))
    assert not result.accepted and result.reason == 'UNOBSERVED_CURVE'


def test_validator_checks_start_state_and_whole_interval_dynamic_bounds():
    path = CatkinSpline(endpoint_controls([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], .1), .1)
    result = validate_spline(path, PathSafety([], observations(), None, DEFAULT_FLIGHT), np.zeros(3))
    assert result.reason.startswith('DYNAMICS_')
    slow = CatkinSpline(path.control_points, 1.)
    result = validate_spline(slow, PathSafety([], observations(), None, DEFAULT_FLIGHT), np.zeros(3),
                             expected_start=np.array([[.1, 0, 0], [0, 0, 0], [0, 0, 0]]))
    assert result.reason == 'START_STATE'


def test_optimizer_produces_bounded_observed_line_and_complete_brake_checks():
    evidence = observations()
    safety = PathSafety([], evidence, None, DEFAULT_FLIGHT)
    field = ObservedESDF([], evidence)
    result = PythonSplineOptimizer(DEFAULT_FLIGHT).optimize(
        [[0, 0, 0], [.8, 0, 0], [1.4, 0, 0]], np.zeros(3), [.2, 0, 0], [.05, 0, 0],
        field, safety)
    assert result.validation.accepted, asdict(result.validation)
    np.testing.assert_allclose(result.path.sample(0.), [[0, 0, 0], [.2, 0, 0], [.05, 0, 0]], atol=1e-9)
    assert result.validation.brake_checks > 5
    assert len(result.attempts) >= 1


def test_optimizer_reserves_dynamics_without_weakening_physical_safety_contract():
    evidence = observations()
    safety = PathSafety([], evidence, None, DEFAULT_FLIGHT)
    limits = SimpleNamespace(max_speed=.7, max_acceleration=.5, max_jerk=2.)
    result = PythonSplineOptimizer(limits).optimize(
        [[0, 0, 0], [.8, 0, 0], [1.4, 0, 0]], np.zeros(3), [.2, 0, 0], [.05, 0, 0],
        ObservedESDF([], evidence), safety)
    assert result.validation.accepted, asdict(result.validation)
    bounds = result.path.derivative_bounds()
    assert bounds['speed'] <= limits.max_speed*(1+1e-7)
    assert bounds['acceleration'] <= limits.max_acceleration*(1+1e-7)
    assert bounds['jerk'] <= limits.max_jerk*(1+1e-7)
    assert safety.flight is DEFAULT_FLIGHT


def test_future_reference_does_not_become_current_body_evidence_exemption():
    actual = np.array([-.2, 0., 0.])
    evidence = observations()
    safety = PathSafety([], evidence, None, DEFAULT_FLIGHT)
    original_known, anchors = safety.known, []

    def known(points, measured, body_padding=.05):
        anchors.append(np.asarray(measured).copy())
        return original_known(points, measured, body_padding)

    safety.known = known
    result = PythonSplineOptimizer(DEFAULT_FLIGHT).optimize(
        [[0, 0, 0], [1.4, 0, 0]], np.zeros(3), [.2, 0, 0], [.05, 0, 0],
        ObservedESDF([], evidence), safety, actual_position=actual)
    assert result.validation.accepted
    assert anchors and all(np.array_equal(anchor, actual) for anchor in anchors)


def test_optimizer_cannot_mark_unknown_route_safe_or_change_invalid_initial_state():
    field = ObservedESDF([], [[0, 0, 0]], bounds=[[-1, -1, -1], [2, 1, 1]])
    safety = PathSafety([], [[0, 0, 0]], None, DEFAULT_FLIGHT)
    optimizer = PythonSplineOptimizer(DEFAULT_FLIGHT)
    result = optimizer.optimize([[0, 0, 0], [1, 0, 0]], np.zeros(3), np.zeros(3), np.zeros(3), field, safety)
    assert not result.validation.accepted
    assert result.validation.reason == 'UNOBSERVED_CURVE'
    result = optimizer.optimize([[0, 0, 0], [1, 0, 0]], np.zeros(3), [1.1, 0, 0], np.zeros(3), field, safety)
    assert result.path is None and result.validation.reason == 'INITIAL_VELOCITY'


def test_safe_curve_is_rejected_when_its_stop_exits_observed_body_volume():
    path = CatkinSpline(endpoint_controls([0, 0, 0], [.5, 0, 0], [0, 0, 0], [1, 0, 0], .6), .6)
    grid = np.meshgrid(np.arange(-1.48, 1.321, .1), np.arange(-1.5, 1.51, .1),
                       np.arange(-.3, .31, .1), indexing='ij')
    evidence = np.stack(grid, axis=-1).reshape(-1, 3)
    safety = PathSafety([], evidence, None, DEFAULT_FLIGHT)
    assert safety.known(path.sample(np.linspace(0, path.duration, 301))[0], np.zeros(3), body_padding=1e-5)
    result = validate_spline(path, safety, np.zeros(3))
    assert result.reason == 'UNOBSERVED_BRAKE' and result.brake_checks > 1


def test_python_optimizer_follows_a_turn_around_observed_obstacle():
    evidence = observations()
    hits = np.array([[1., 0., 0.]])
    field = ObservedESDF(hits, evidence)
    safety = PathSafety(hits, evidence, None, DEFAULT_FLIGHT)
    result = PythonSplineOptimizer(DEFAULT_FLIGHT).optimize(
        [[0, 0, 0], [0, 1.2, 0], [2, 1.2, 0], [2, 0, 0]],
        np.zeros(3), np.zeros(3), np.zeros(3), field, safety)
    assert result.validation.accepted, asdict(result.validation)
    points = result.path.sample(np.linspace(0., result.path.duration, 401))[0]
    assert points[:, 1].max() > .8
    assert safety.clear(points)


def test_constrained_refinement_preserves_clearance_and_exact_moving_endpoint_state():
    evidence = observations()
    hits = np.array([[1., 0., 0.]])
    safety = PathSafety(hits, evidence, None, DEFAULT_FLIGHT)
    limits = SimpleNamespace(max_speed=.5, max_acceleration=.6, max_jerk=2.6)
    start = np.array([[0., 0., 0.], [0., .15, 0.], [0., .03, 0.]])
    result = PythonSplineOptimizer(limits, retimings=6, enforce_feasible_controls=True).optimize(
        [[0, 0, 0], [0, 1.2, 0], [2, 1.2, 0], [2, 0, 0]], *start,
        ObservedESDF(hits, evidence), safety)
    assert result.validation.accepted, asdict(result.validation)
    np.testing.assert_allclose(result.path.sample(0.), start, atol=1e-9)
    bounds = result.path.derivative_bounds()
    assert bounds['speed'] <= .5*(1+1e-7)
    assert bounds['acceleration'] <= .6*(1+1e-7)
    assert bounds['jerk'] <= 2.6*(1+1e-7)
    assert result.attempts[-1]['constraint_refinement'] is not None
    assert safety.clear(result.path.sample(np.linspace(0, result.path.duration, 401))[0])


@pytest.mark.parametrize('length,start', [(4., 0.), (4., .9), (.3, 0.), (.5, .9)])
def test_stop_profile_starts_at_the_given_speed_and_rests_at_the_route_end(length, start):
    duration, fraction = stop_profile(length, start, 1., .3)
    tau = np.linspace(0., 1., 2001)
    s = fraction(tau)*length
    speed = np.diff(s)/(np.diff(tau)*duration)
    assert s[0] == 0. and np.isclose(s[-1], length) and (np.diff(s) >= -1e-12).all()
    assert speed.max() <= 1.+1e-6 and abs(speed[0]-start) < .01 and speed[-1] < .01


def test_profile_timing_keeps_cruising_where_uniform_timing_already_brakes():
    evidence = observations()
    safety = PathSafety([], evidence, None, DEFAULT_FLIGHT)
    limits = SimpleNamespace(max_speed=1., max_acceleration=.6, max_jerk=2.6)
    start = np.array([[0., 0., 0.], [.9, 0., 0.], [0., 0., 0.]])
    speeds = {}
    for profile in (None, .3):
        result = PythonSplineOptimizer(limits, retimings=6, enforce_feasible_controls=True,
                                       profile_acceleration=profile).optimize(
            [[0, 0, 0], [3.4, 0, 0]], *start, ObservedESDF([], evidence), safety, nominal_speed=1.)
        assert result.validation.accepted and len(result.attempts) == 1, asdict(result.validation)
        np.testing.assert_allclose(result.path.sample(0.), start, atol=1e-9)
        speeds[profile] = np.linalg.norm(result.path.sample(.5)[1])
    # One replanning period (~0.5 s) is all that runs of each spline.
    assert speeds[.3] >= .9 > .85 > speeds[None]


def test_profile_acceleration_must_fit_the_planning_limit():
    limits = SimpleNamespace(max_speed=1., max_acceleration=.6, max_jerk=2.6)
    for value in (0., .7):
        with pytest.raises(ValueError, match='profile_acceleration'):
            PythonSplineOptimizer(limits, profile_acceleration=value)


def blind_band_observations():
    """Free space seen while stopped: body layers ahead, the lower layer only
    inside the downward cone (between it and the lowest MID-360 ray)."""
    xs, ys = np.arange(-1.6, 4.01, .2), np.arange(-2., 2.01, .2)
    upper = np.stack(np.meshgrid(xs, ys, np.arange(0., 1.01, .2), indexing='ij'), axis=-1).reshape(-1, 3)
    lower = np.stack(np.meshgrid(xs, ys, [-.4, -.2], indexing='ij'), axis=-1).reshape(-1, 3)
    return np.vstack([upper, lower[np.hypot(lower[:, 0], lower[:, 1]) <= .6]])


@pytest.mark.parametrize('branch', [(0., .9), (0., 1.), (1., .6), (1.1, .3)])
def test_clearance_cost_does_not_push_a_stopped_body_under_its_observed_layer(branch):
    # Run cc69e994 stalled for 85 s on this; a branch 0.9 m overhead lowered
    # the curve 0.6 m. The cylinder check clears all of these at route height.
    hits = np.c_[np.arange(-.5, 3.51, .1), np.full(41, branch[0]), np.full(41, branch[1])]
    evidence = blind_band_observations()
    evidence = evidence[np.linalg.norm(evidence[:, None]-hits[None], axis=2).min(axis=1) > .15]
    safety = PathSafety(hits, evidence, None, DEFAULT_FLIGHT).with_clearance_reserve(.15)
    route = [[0., 0., 0.], [1.5, 0., 0.], [3., 0., 0.]]
    assert safety.clear(route) and safety.known(route, np.zeros(3))
    result = PythonSplineOptimizer(DEFAULT_FLIGHT).optimize(
        route, np.zeros(3), np.zeros(3), np.zeros(3), ObservedESDF(hits, evidence), safety)
    assert result.validation.accepted, result.validation.reason
    heights = result.path.sample(np.linspace(0., result.path.duration, 200))[0][:, 2]
    assert heights.min() > -.03


def flat_band(high, low=-.8):
    from omni_drones.navigation.forest.planning import HeightBand
    xs, ys = np.arange(-3., 6.01, .25), np.arange(-3., 3.01, .25)
    return HeightBand(xs, ys, np.full((len(xs), len(ys)), low), np.full((len(xs), len(ys)), high))


def test_a_height_band_lets_points_come_back_but_not_leave():
    band = flat_band(.3, low=-.5)
    assert band.leaving([[0, 0, 0], [1, 0, .2]]) and not band.leaving([[0, 0, 0], [1, 0, .4]])
    assert band.leaving([[0, 0, .6], [.5, 0, .45], [1, 0, .2]])
    assert not band.leaving([[0, 0, .6], [.5, 0, .7]]) and not band.leaving([[0, 0, .6], [.5, 0, .2], [1, 0, .4]])
    low, high = band.limits([[10., 0.]])
    assert np.isneginf(low[0]) and np.isposinf(high[0])
    import pickle  # it travels to the spline worker process with the planning safety
    assert not pickle.loads(pickle.dumps(band)).leaving([[0, 0, 0], [1, 0, .4]])


@pytest.mark.parametrize('climb, ceiling, reason', [(.3, .26, 'HEIGHT_BAND_CURVE'), (.3, .28, 'HEIGHT_BAND_BRAKE'),
                                                   (.1, .26, ''), (.1, .28, '')])
def test_a_spline_handed_on_while_climbing_and_its_stops_stay_under_the_ceiling(climb, ceiling, reason):
    # Run 04376dbe: the route stayed at the band's 3.35 m, but each spline took
    # over the climb's upward speed and the reference reached 3.58 m.
    evidence = observations()
    safety = PathSafety([], evidence, None, DEFAULT_FLIGHT)
    safety.height_band = flat_band(ceiling)
    result = PythonSplineOptimizer(DEFAULT_FLIGHT).optimize(
        [[0, 0, 0], [.8, 0, .25], [1.8, 0, .25]], np.zeros(3), [.3, 0, climb], np.zeros(3),
        ObservedESDF([], evidence), safety)
    assert result.validation.accepted is (not reason) and result.validation.reason == reason
    if not reason:
        assert result.path.sample(np.linspace(0., result.path.duration, 200))[0][:, 2].max() <= ceiling
