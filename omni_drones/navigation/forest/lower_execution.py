"""Measured-state evidence and one-shot stopping helpers for the lower executor.

This module has no simulator dependency. A stop command is a reference to execute;
neither constructing it nor passing its nominal checks proves the vehicle stopped.
The caller owns command-clock latching and any persistent task failure status.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .local import BODY_NUMERIC_TOLERANCE
from .spline_runtime import brake_from_reference
from .trajectory import QuinticPath


def _xyz(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite world XYZ vector')
    return result


def _time(value, previous=None):
    if not np.isscalar(value) or not math.isfinite(value):
        raise ValueError('simulation time must be finite')
    result = float(value)
    if previous is not None and result < previous:
        raise ValueError('simulation time must be monotonic')
    return result


def _positive(value, name):
    if not np.isscalar(value) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


class ActualAccelerationEstimator:
    """Time-weighted mean of the last ``samples`` actual-velocity differences.

    ``update`` always returns an XYZ array. Before a positive-time velocity pair
    exists it returns a zero *placeholder* with ``ready=False``; the caller must
    not relabel that placeholder as measured acceleration evidence. Repeated
    timestamps retain the estimate and baseline, including during simulator pause.
    """
    def __init__(self, samples=5):
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError('samples must be a positive integer')
        self.samples = samples
        self._differences = deque(maxlen=samples)
        self._time, self._velocity = None, None
        self._estimate = np.zeros(3)
        self._duplicates = 0

    @property
    def ready(self):
        return bool(self._differences)

    @property
    def metadata(self):
        duration = sum(row[1] for row in self._differences)
        return dict(source='actual_world_velocity_finite_difference',
                    averaging='time_weighted_last_intervals', clock='simulation',
                    ready=self.ready, intervals=len(self._differences),
                    maximum_intervals=self.samples, window_seconds=float(duration),
                    window_start=None if self._time is None else self._time-duration,
                    timestamp=self._time, ignored_duplicate_timestamps=self._duplicates,
                    placeholder=not self.ready)

    def update(self, now, velocity):
        now = _time(now, self._time)
        velocity = _xyz(velocity, 'actual velocity')
        if self._time is None:
            self._time, self._velocity = now, velocity.copy()
            return self._estimate.copy()
        if now == self._time:
            self._duplicates += 1
            return self._estimate.copy()
        self._differences.append((velocity-self._velocity, now-self._time))
        duration = sum(row[1] for row in self._differences)
        self._estimate = np.sum([row[0] for row in self._differences], axis=0)/duration
        self._time, self._velocity = now, velocity.copy()
        return self._estimate.copy()


class MeasuredStopMonitor:
    """Require continuous measured/reference rest plus a stable measured position.

    The monitor reports current stop evidence only. It does not set task status,
    clear a fault, or turn a failed task into success. Motion after a detected stop
    clears ``stopped`` again; the caller may end execution when evidence is true.
    """
    def __init__(self, speed_tolerance=.05, acceleration_tolerance=.15, dwell=.3,
                 position_tolerance=.05):
        self.speed_tolerance = _positive(speed_tolerance, 'speed tolerance')
        self.acceleration_tolerance = _positive(acceleration_tolerance, 'acceleration tolerance')
        self.dwell = _positive(dwell, 'dwell')
        self.position_tolerance = _positive(position_tolerance, 'position tolerance')
        self._last_time = None
        self._since, self._anchor = None, None
        self._stopped = False

    @property
    def stopped(self):
        return self._stopped

    @property
    def dwell_seconds(self):
        return 0. if self._since is None else self._last_time-self._since

    def update(self, now, position, velocity, acceleration, reference_velocity,
               reference_acceleration):
        now = _time(now, self._last_time)
        p, v, a, rv, ra = [_xyz(value, name) for value, name in (
            (position, 'actual position'), (velocity, 'actual velocity'),
            (acceleration, 'estimated actual acceleration'),
            (reference_velocity, 'reference velocity'), (reference_acceleration, 'reference acceleration'))]
        self._last_time = now
        quiet = (np.linalg.norm(v) <= self.speed_tolerance
                 and np.linalg.norm(rv) <= self.speed_tolerance
                 and np.linalg.norm(a) <= self.acceleration_tolerance
                 and np.linalg.norm(ra) <= self.acceleration_tolerance)
        if not quiet:
            self._since, self._anchor, self._stopped = None, None, False
            return False
        if self._anchor is None or np.linalg.norm(p-self._anchor) > self.position_tolerance:
            self._since, self._anchor = now, p.copy()
        self._stopped = now-self._since >= self.dwell-1e-12
        return self._stopped


@dataclass
class StopPlan:
    command: QuinticPath
    normal_certified: bool
    reason: str
    nominal_envelope: np.ndarray
    actual_envelope: np.ndarray
    normal_violation: bool
    recovery_geometric: bool = False
    recovery_acceleration_limit: float | None = None
    violations: tuple[str, ...] = ()

    def audit(self):
        """Small JSON-compatible evidence; reference data stay on the command."""
        return dict(normal_certified=self.normal_certified, reason=self.reason,
                    normal_violation=self.normal_violation, violations=list(self.violations),
                    recovery_geometric=self.recovery_geometric,
                    recovery_acceleration_limit=self.recovery_acceleration_limit,
                    nominal_envelope=self.nominal_envelope.tolist(),
                    actual_envelope=self.actual_envelope.tolist())


def _recovery_brake(position, velocity, acceleration, flight):
    # This bound only allows constructing a geometric recovery hypothesis from
    # an already excessive initial acceleration. It is not a verified actuator
    # limit or permission to count normal-limit violations as certified motion.
    limit = max(float(flight.max_acceleration), 1.01*float(np.linalg.norm(acceleration)))
    return QuinticPath.brake(position, velocity, acceleration, limit, flight.max_jerk), limit


def _geometry_clear(safety, nominal, actual_envelope, position):
    if safety is None or not len(actual_envelope):
        return False
    # Check separate polylines; concatenation would invent an unrelated segment
    # from the reference stopping endpoint back to the measured vehicle body.
    return all(safety.clear(points) and safety.known(points, position,
               body_padding=BODY_NUMERIC_TOLERANCE) for points in (nominal, actual_envelope))


def start_stop(command, elapsed, safety, *, actual_position, actual_velocity,
               actual_acceleration, flight=None, reference_flight=None):
    """Construct one continuous reference stop without changing any command clock.

    Always execute/latch the returned ``command`` only through the caller's
    existing controller and observe the eventual physical stop. If measured
    acceleration already exceeds normal limits, a recovery envelope may still be
    geometrically checked using an explicitly larger construction bound; its
    ``normal_certified`` remains false. Calling this repeatedly is *not* latching.
    ``recovery_geometric`` retains the existing nominal-body sampled evidence
    contract; it is not a bound on the actual tilted body or tracking-error tube.
    """
    elapsed = _time(elapsed)
    if elapsed < 0:
        raise ValueError('elapsed must be nonnegative')
    actual = _xyz(actual_position, 'actual position')
    actual_v = _xyz(actual_velocity, 'actual velocity')
    actual_a = _xyz(actual_acceleration, 'estimated actual acceleration')
    flight = safety.flight if safety is not None else flight
    if flight is None:
        raise ValueError('normal flight limits are required')
    acceleration_limit = _positive(flight.max_acceleration, 'normal acceleration limit')
    _positive(flight.max_jerk, 'normal jerk limit')
    speed_limit = _positive(flight.max_speed, 'normal speed limit')
    p, v, a = [_xyz(value, name) for value, name in zip(command.sample(elapsed),
              ('reference position', 'reference velocity', 'reference acceleration'))]
    violations = tuple(name for name, value, limit in (
        ('ACTUAL_ACCELERATION', actual_a, acceleration_limit),
        ('ACTUAL_SPEED', actual_v, speed_limit),
        ('REFERENCE_ACCELERATION', a, acceleration_limit),
        ('REFERENCE_SPEED', v, speed_limit)) if np.linalg.norm(value) > limit*1.001)
    try:
        brake, checked = brake_from_reference(command, elapsed, safety,
            actual_position=actual, actual_velocity=actual_v, actual_acceleration=actual_a, flight=flight,
            reference_flight=reference_flight)
        nominal, actual_envelope = checked.nominal_envelope, checked.actual_brake_envelope
        reason = checked.reason
        normal = checked.accepted and not violations
        if checked.accepted and violations:
            reason = 'NORMAL_'+violations[0]
    except ValueError:
        # The old reference itself may also be outside its normal budget. Keep
        # its exact p/v/a boundary, but explicitly classify recovery as abnormal.
        brake, _ = _recovery_brake(p, v, a, flight)
        nominal, actual_envelope = brake.remaining(0., actual), np.empty((0, 3))
        reason, normal = 'REFERENCE_BRAKE_DYNAMICS', False
    recovery_limit = None
    if not len(actual_envelope):
        recovery, recovery_limit = _recovery_brake(actual, actual_v, actual_a, flight)
        actual_envelope = recovery.remaining(0., actual)
    geometric = _geometry_clear(safety, nominal, actual_envelope, actual)
    return StopPlan(brake, bool(normal), reason, nominal, actual_envelope,
                    bool(violations), geometric, recovery_limit, violations)
