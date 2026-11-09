"""Regression tests for asynchronous clock handoff and independent stops."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from omni_drones.navigation.forest.catkin_spline import CatkinSpline
from omni_drones.navigation.forest.local import VelocityTrajectory
from omni_drones.navigation.forest.spline_optimizer import endpoint_controls
from omni_drones.navigation.forest.spline_runtime import (
    ExecutableSpline, activation_check, brake_from_reference, execution_check,
    make_handoff, observation_age, prepare_candidate,
)
from omni_drones.navigation.forest.spline_flight import anchored_route


FLIGHT = SimpleNamespace(max_speed=1., max_acceleration=1., max_jerk=4.)


def test_future_anchor_preserves_remaining_turn_without_returning_to_passed_vertices():
    route = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]]
    np.testing.assert_allclose(anchored_route(route, [1, .4, 0]), [[1, .4, 0], [1, 1, 0], [2, 1, 0]])
    np.testing.assert_allclose(anchored_route(route, [2, 1, 0]), [[2, 1, 0]])


class Safety:
    flight = FLIGHT

    def __init__(self):
        self.polylines = []

    def clear(self, points, start=None):
        self.polylines.append(np.asarray(points))
        return True

    def clear_points(self, points, padding=0.):
        return np.ones(len(points), bool)

    def clear_leaving(self, points, padding, start, **kwargs):
        return bool(self.clear_points(points, padding).all())

    def known(self, points, actual, body_padding=0.):
        return True


def setup_candidate():
    command = VelocityTrajectory(np.zeros(3), [.1, 0., 0.], np.zeros(3), [.2, 0., 0.], FLIGHT)
    request = make_handoff(command, 0., .1, .1, 10., 2, 3, activation_lead=.2, control_dt=.01)
    p, v, a = request.start_pva
    spline = CatkinSpline(endpoint_controls(p, v, a, [1., 0., 0.], 1.), 1.)
    candidate = prepare_candidate(spline, request, Safety(), command.sample(.1)[0])
    assert candidate.validation.accepted
    return command, candidate


def check(candidate, command, now, **kwargs):
    p, v, a = command.sample(now)
    return activation_check(candidate, command, 0., now, 10.2, 2, 3, Safety(),
        actual_position=p, actual_velocity=v, actual_acceleration=a,
        deadline=.3, max_observation_age=.3, horizon=.2,
        position_tolerance=.1, velocity_tolerance=.1, **kwargs)


def test_activation_only_at_promised_tick_and_keeps_exact_pva():
    command, candidate = setup_candidate()
    assert check(candidate, command, .29).reason == 'WAIT'
    assert check(candidate, command, .3).accepted
    assert check(candidate, command, .31).reason == 'MISSED_SWITCH'
    np.testing.assert_allclose(candidate.path.sample(0.), command.sample(.3), atol=1e-12)
    assert candidate.request.switch_time == pytest.approx(.3)


def test_superseded_goal_or_brake_generation_cannot_replace_command():
    command, candidate = setup_candidate()
    for request in (replace(candidate.request, generation=1), replace(candidate.request, revision=2)):
        assert check(replace(candidate, request=request), command, .3).reason == 'SUPERSEDED'
    changed = VelocityTrajectory([.1, 0., 0.], [.1, 0., 0.], np.zeros(3), [.2, 0., 0.], FLIGHT)
    assert check(candidate, changed, .3).reason == 'START_STATE'


def test_freshness_accounts_for_both_clocks_and_rejects_reversal():
    assert observation_age(.3, 10.05, .1, 10.) == pytest.approx(.2)
    assert observation_age(.15, 10.2, .1, 10.) == pytest.approx(.2)
    for clocks in ((.09, 10.2, .1, 10.), (.3, 9.9, .1, 10.)):
        with pytest.raises(ValueError):
            observation_age(*clocks)
    command, candidate = setup_candidate()
    stale = replace(candidate, request=replace(candidate.request, captured=9.))
    assert check(stale, command, .3).reason == 'STALE'


def test_executable_snapshot_is_immutable_and_holds_after_endpoint():
    source = CatkinSpline(endpoint_controls([0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], 1.), 1.)
    path = ExecutableSpline(source)
    source.control_points[:] = 9.
    np.testing.assert_allclose(path.sample(path.duration+2.), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        path.sample(-.01)
    with pytest.raises(ValueError):
        path.spline.control_points[0] = 5.
    # Curved interiors are sampled using the global speed bound even at zero-speed endpoints.
    assert len(path.remaining(0., [0, 0, 0], path.duration)) > 10


def test_runtime_checks_measured_tracking_and_both_separate_braking_paths():
    command, candidate = setup_candidate()
    path = candidate.path
    p, v, a = path.sample(0.)
    safety = Safety()
    result = execution_check(path, 0., safety, actual_position=p, actual_velocity=v,
        actual_acceleration=a, horizon=.2, position_tolerance=.1, velocity_tolerance=.1)
    assert result.accepted
    assert len(safety.polylines) == 2
    np.testing.assert_array_equal(safety.polylines[0], result.nominal_envelope)
    np.testing.assert_array_equal(safety.polylines[1], result.actual_brake_envelope)
    for position, velocity, reason in ((p+[.2, 0, 0], v, 'TRACKING_POSITION'),
                                       (p, v+[.2, 0, 0], 'TRACKING_VELOCITY')):
        result = execution_check(path, 0., safety, actual_position=position, actual_velocity=velocity,
            actual_acceleration=a, horizon=.2, position_tolerance=.1, velocity_tolerance=.1)
        assert result.reason == reason


def test_actual_stop_unknown_is_rejected_and_brake_does_not_reset_reference():
    command, _ = setup_candidate()
    p, v, a = command.sample(.3)
    safety = Safety()
    calls = []

    def known(points, actual, body_padding):
        calls.append(points)
        return len(calls) % 2 == 1

    safety.known = known
    brake, result = brake_from_reference(command, .3, safety,
        actual_position=p, actual_velocity=v, actual_acceleration=a)
    assert result.reason == 'ACTUAL_UNOBSERVED'
    np.testing.assert_allclose(brake.sample(0.), [p, v, a], atol=1e-12)
    _, result = brake_from_reference(command, .3, None, actual_position=p,
        actual_velocity=v, actual_acceleration=a, flight=FLIGHT)
    assert result.reason == 'NO_OBSERVATION'


def test_execution_lease_checks_the_longer_reserved_reference_stop():
    command = VelocityTrajectory(np.zeros(3), [.4, 0., 0.], np.zeros(3), [.4, 0., 0.], FLIGHT)
    kwargs = dict(actual_position=np.zeros(3), actual_velocity=[.4, 0., 0.],
                  actual_acceleration=np.zeros(3), horizon=.2, position_tolerance=.1, velocity_tolerance=.1)
    normal = execution_check(command, 0., Safety(), **kwargs)
    reserved = SimpleNamespace(max_acceleration=.6, max_jerk=2.6)
    cautious = execution_check(command, 0., Safety(), **kwargs, reference_flight=reserved)
    assert normal.accepted and cautious.accepted
    assert cautious.nominal_envelope[:, 0].max() > normal.nominal_envelope[:, 0].max()+.05
    np.testing.assert_allclose(cautious.actual_brake_envelope, normal.actual_brake_envelope)
