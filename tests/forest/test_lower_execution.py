"""Actual-state evidence and honest stopping classification without Isaac."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from omni_drones.navigation.forest.local import VelocityTrajectory
from omni_drones.navigation.forest.lower_execution import (
    ActualAccelerationEstimator, MeasuredStopMonitor, start_stop,
)


FLIGHT = SimpleNamespace(max_speed=1., max_acceleration=1., max_jerk=4.)
ZERO = np.zeros(3)


class Safety:
    flight = FLIGHT

    def __init__(self, known=True, clear=True):
        self.is_known, self.is_clear = known, clear
        self.checked = []

    def clear(self, points):
        self.checked.append(np.asarray(points).copy())
        return self.is_clear

    def known(self, points, actual, body_padding=0.):
        return self.is_known


def test_acceleration_uses_actual_velocity_and_weights_irregular_time_intervals():
    estimator = ActualAccelerationEstimator(samples=2)
    np.testing.assert_array_equal(estimator.update(0., ZERO), ZERO)
    assert not estimator.ready and estimator.metadata['placeholder']
    np.testing.assert_allclose(estimator.update(.1, [.1, 0, 0]), [1., 0, 0])
    np.testing.assert_allclose(estimator.update(.3, [.5, 0, 0]), [5/3, 0, 0])
    np.testing.assert_allclose(estimator.update(.4, [.8, 0, 0]), [7/3, 0, 0])
    assert estimator.metadata['window_seconds'] == pytest.approx(.3)
    assert estimator.metadata['window_start'] == pytest.approx(.1)
    assert estimator.ready and estimator.metadata['intervals'] == 2
    json.dumps(estimator.metadata)


def test_paused_clock_does_not_create_or_pollute_acceleration_estimate():
    estimator = ActualAccelerationEstimator()
    estimator.update(0., ZERO)
    np.testing.assert_array_equal(estimator.update(0., [100, 0, 0]), ZERO)
    assert not estimator.ready
    value = estimator.update(.1, [.1, 0, 0])
    value[:] = 500.
    np.testing.assert_allclose(estimator.update(.1, [100, 0, 0]), [1., 0, 0])
    np.testing.assert_allclose(estimator.update(.2, [.2, 0, 0]), [1., 0, 0])
    assert estimator.metadata['ignored_duplicate_timestamps'] == 2


@pytest.mark.parametrize('samples', [0, -1, True, 1.5])
def test_invalid_acceleration_window_rejected(samples):
    with pytest.raises(ValueError):
        ActualAccelerationEstimator(samples)


def test_acceleration_rejects_reversed_nonfinite_or_wrong_shape_without_mutation():
    estimator = ActualAccelerationEstimator()
    estimator.update(1., ZERO)
    for time, velocity in ((.9, ZERO), (np.nan, ZERO), (2., [0, 0]), (2., [0, np.inf, 0])):
        with pytest.raises(ValueError):
            estimator.update(time, velocity)
    assert estimator.metadata['timestamp'] == 1.
    assert not estimator.ready


def test_stop_requires_actual_reference_rest_and_continuous_position_dwell():
    monitor = MeasuredStopMonitor()
    assert not monitor.update(0., ZERO, [.2, 0, 0], ZERO, ZERO, ZERO)
    assert not monitor.update(1., ZERO, ZERO, ZERO, [.2, 0, 0], ZERO)
    assert not monitor.update(2., ZERO, ZERO, [.3, 0, 0], ZERO, ZERO)
    assert not monitor.update(3., ZERO, ZERO, ZERO, ZERO, [.3, 0, 0])
    assert not monitor.update(4., ZERO, ZERO, ZERO, ZERO, ZERO)
    assert not monitor.update(4.2, [.06, 0, 0], ZERO, ZERO, ZERO, ZERO)
    assert not monitor.update(4.49, [.06, 0, 0], ZERO, ZERO, ZERO, ZERO)
    assert monitor.update(4.5, [.06, 0, 0], ZERO, ZERO, ZERO, ZERO)
    assert monitor.stopped
    assert not monitor.update(4.6, [.06, 0, 0], [.2, 0, 0], ZERO, ZERO, ZERO)
    assert not monitor.stopped and monitor.dwell_seconds == 0.


def test_stop_clock_pause_cannot_advance_dwell_and_invalid_input_is_rejected():
    monitor = MeasuredStopMonitor()
    for _ in range(10):
        assert not monitor.update(1., ZERO, ZERO, ZERO, ZERO, ZERO)
    assert monitor.dwell_seconds == 0.
    with pytest.raises(ValueError):
        monitor.update(.9, ZERO, ZERO, ZERO, ZERO, ZERO)
    with pytest.raises(ValueError):
        monitor.update(2., [0, np.nan, 0], ZERO, ZERO, ZERO, ZERO)
    assert not monitor.stopped


def active_command():
    return VelocityTrajectory(ZERO, [.3, 0, 0], [.1, 0, 0], [.2, 0, 0], FLIGHT)


def test_normal_reference_stop_preserves_exact_pva_and_checks_separate_envelopes():
    command, safety = active_command(), Safety()
    p, v, a = command.sample(.2)
    result = start_stop(command, .2, safety, actual_position=p, actual_velocity=v, actual_acceleration=a)
    assert result.normal_certified and not result.normal_violation
    assert result.reason == '' and result.recovery_geometric
    np.testing.assert_allclose(result.command.sample(0.), [p, v, a], atol=1e-12)
    end = result.command.sample(result.command.duration)
    np.testing.assert_allclose(end[1:], [ZERO, ZERO], atol=1e-12)
    assert len(safety.checked) >= 2
    assert len(result.nominal_envelope) > 1 and len(result.actual_envelope) > 1
    json.dumps(result.audit())


def test_excessive_actual_acceleration_keeps_reference_stop_but_never_certifies():
    command, safety = active_command(), Safety()
    p, v, a = command.sample(.2)
    result = start_stop(command, .2, safety, actual_position=p, actual_velocity=v,
                        actual_acceleration=[1.2, 0, 0])
    assert not result.normal_certified and result.normal_violation
    assert result.reason == 'ACTUAL_BRAKE_DYNAMICS'
    assert result.recovery_geometric
    assert result.recovery_acceleration_limit > 1.2
    assert result.violations == ('ACTUAL_ACCELERATION',)
    np.testing.assert_allclose(result.command.sample(0.), [p, v, a], atol=1e-12)
    assert len(result.actual_envelope) > 2
    assert safety.flight.max_acceleration == 1.


def test_initial_speed_violation_cannot_be_certified_by_a_feasible_braking_shape():
    command = VelocityTrajectory(ZERO, [1.2, 0., 0.], ZERO, [1.2, 0., 0.], FLIGHT)
    result = start_stop(command, 0., Safety(), actual_position=ZERO,
                        actual_velocity=[1.2, 0., 0.], actual_acceleration=ZERO)
    assert result.normal_violation and not result.normal_certified
    assert result.recovery_geometric
    assert 'ACTUAL_SPEED' in result.violations and 'REFERENCE_SPEED' in result.violations


@pytest.mark.parametrize('known,clear,reason', [(False, True, 'REFERENCE_UNOBSERVED'),
                                               (True, False, 'REFERENCE_OBSTACLE')])
def test_unknown_or_colliding_stop_is_not_a_normal_or_geometric_certificate(known, clear, reason):
    command, safety = active_command(), Safety(known, clear)
    p, v, a = command.sample(.2)
    result = start_stop(command, .2, safety, actual_position=p, actual_velocity=v, actual_acceleration=a)
    assert not result.normal_certified and not result.recovery_geometric
    assert result.reason == reason


def test_unobserved_fallback_returns_command_without_claiming_certification():
    command = active_command()
    p, v, a = command.sample(.2)
    result = start_stop(command, .2, None, actual_position=p, actual_velocity=v,
                        actual_acceleration=a, flight=FLIGHT)
    assert result.reason == 'NO_OBSERVATION'
    assert not result.normal_certified and not result.recovery_geometric
    assert len(result.actual_envelope) > 1


def test_reference_stop_reserves_margin_while_actual_state_uses_normal_limits():
    from copy import copy
    command = VelocityTrajectory(ZERO, [.5, 0., 0.], ZERO, [.5, 0., 0.], FLIGHT)
    safety = Safety()
    reserved = copy(FLIGHT)
    reserved.max_acceleration = .8
    reserved.max_jerk = 3.
    normal = start_stop(command, 0., safety, actual_position=ZERO,
                        actual_velocity=[.5, 0., 0.], actual_acceleration=ZERO)
    result = start_stop(command, 0., safety, actual_position=ZERO,
        actual_velocity=[.5, 0., 0.], actual_acceleration=[.9, 0., 0.], reference_flight=reserved)
    assert result.normal_certified and not result.normal_violation
    assert result.command.duration > normal.command.duration
    times = np.linspace(0., result.command.duration, 1001)
    assert np.linalg.norm(result.command.sample(times)[2], axis=1).max() <= .8*1.001
    assert safety.flight.max_acceleration == 1.


def test_reserved_reference_cannot_hide_excessive_measured_acceleration():
    from copy import copy
    reserved = copy(FLIGHT)
    reserved.max_acceleration = .8
    reserved.max_jerk = 3.
    command = VelocityTrajectory(ZERO, [.5, 0., 0.], ZERO, [.5, 0., 0.], FLIGHT)
    result = start_stop(command, 0., Safety(), actual_position=ZERO,
        actual_velocity=[.5, 0., 0.], actual_acceleration=[1.2, 0., 0.], reference_flight=reserved)
    assert not result.normal_certified and result.normal_violation
    assert 'ACTUAL_ACCELERATION' in result.violations


def test_abnormal_reference_still_gets_continuous_recovery_with_explicit_violation():
    class Command:
        def sample(self, t):
            return ZERO, np.array([.3, 0, 0]), np.array([1.2, 0, 0])

    result = start_stop(Command(), 0., Safety(), actual_position=ZERO,
                        actual_velocity=[.3, 0, 0], actual_acceleration=[1.2, 0, 0])
    assert not result.normal_certified and result.normal_violation
    assert result.reason == 'REFERENCE_BRAKE_DYNAMICS'
    assert 'REFERENCE_ACCELERATION' in result.violations
    np.testing.assert_allclose(result.command.sample(0.), Command().sample(0.), atol=1e-12)
