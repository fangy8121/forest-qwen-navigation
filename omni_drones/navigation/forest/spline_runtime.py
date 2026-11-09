"""Clock and safety adapters for executing validated splines in the simulator.

This module does not run a controller or a worker. The caller keeps executing
the old command until the fixed handoff time, then sends ``path.sample(t)``'s
position, velocity and acceleration unchanged to the position controller.
Every command replacement (including braking and a new goal) must increment
the caller's generation. A rejected candidate never resets the active clock.
"""
from dataclasses import dataclass, field
import numpy as np

from .catkin_spline import CatkinSpline
from .local import BODY_NUMERIC_TOLERANCE
from .spline_optimizer import SplineValidation, validate_spline
from .trajectory import QuinticPath


def _xyz(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite world XYZ')
    return result


def _nonnegative(value, name):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


class ExecutableSpline:
    """Own a spline snapshot and hold its stationary endpoint after completion.

    The numeric CatkinSpline contract stays strict; only this execution adapter
    permits times beyond its duration. Negative times remain an error so an
    early activation cannot silently clamp to the first command.
    """
    def __init__(self, spline):
        self.spline = CatkinSpline(spline.control_points, spline.interval)
        _, velocity, acceleration = self.spline.sample(self.spline.duration)
        if np.linalg.norm(velocity)+np.linalg.norm(acceleration) > 1e-8:
            raise ValueError('An executable spline must end with zero velocity and acceleration')
        self.duration = self.spline.duration
        self.speed_bound = self.spline.derivative_bounds()['speed']
        for array in (self.spline.control_points, self.spline.curve.c,
                      *(curve.c for curve in self.spline.derivatives)):
            array.setflags(write=False)

    def sample(self, t):
        t = np.asarray(t, dtype=float)
        if not np.isfinite(t).all() or (t < 0).any():
            raise ValueError('Execution time must be finite and nonnegative')
        p, v, a = self.spline.sample(np.minimum(t, self.duration))
        # Suppress floating point endpoint residue without changing the curve.
        return p, np.where((t >= self.duration)[..., None], 0., v), np.where(
            (t >= self.duration)[..., None], 0., a)

    def remaining(self, t, actual, horizon=None):
        t = _nonnegative(t, 'elapsed')
        actual = _xyz(actual, 'actual position')
        horizon = max(0., self.duration-t) if horizon is None else _nonnegative(horizon, 'horizon')
        end = min(self.duration, t+horizon)
        start = min(t, self.duration)
        # Bound chord length using the global derivative hull, not endpoint speed.
        count = max(2, int(np.ceil((end-start)*max(self.speed_bound, 1e-6)/.02))+1)
        return np.vstack([actual, self.sample(np.linspace(start, end, count))[0]])

    def as_dict(self):
        return self.spline.as_dict()


@dataclass(frozen=True)
class SplineHandoff:
    switch_time: float
    source_time: float
    captured: float
    generation: int
    revision: int
    start_pva: np.ndarray


def make_handoff(command, command_start, now, source_time, captured, generation,
                 revision, *, activation_lead, control_dt):
    """Freeze the *reference* state at a future controller tick, before work.

    ``source_time`` is the scan's simulation time; ``captured`` is the monotonic
    wall time at scan capture. Neither is replaced by worker completion time.
    """
    values = np.array([command_start, now, source_time, captured, activation_lead, control_dt])
    if (not np.isfinite(values).all() or control_dt <= 0 or activation_lead <= 0
            or now < command_start-1e-9 or source_time > now+1e-9):
        raise ValueError('Invalid command/observation clocks or activation lead')
    # Quantize relative to this control tick, avoiding a global epoch assumption.
    switch = float(now+np.ceil(activation_lead/control_dt-1e-9)*control_dt)
    state = np.asarray(command.sample(switch-command_start), dtype=float).copy()
    if state.shape != (3, 3) or not np.isfinite(state).all():
        raise ValueError('Active command produced an invalid reference state')
    state.setflags(write=False)
    return SplineHandoff(switch, float(source_time), float(captured),
                         generation, revision, state)


@dataclass(frozen=True)
class SplineCandidate:
    request: SplineHandoff
    path: ExecutableSpline | None
    validation: SplineValidation


def prepare_candidate(spline, request, safety, actual_position):
    """Worker-side full curve and stopping validation using measured position.

    A future reference anchor must never be used as the current-body evidence
    exemption. This repeats the optimizer's validation deliberately: historical
    optimizers used their reference start as ``actual``.
    """
    actual = _xyz(actual_position, 'actual position')
    snapshot = CatkinSpline(spline.control_points, spline.interval)
    validation = validate_spline(snapshot, safety, actual, expected_start=request.start_pva)
    return SplineCandidate(request, ExecutableSpline(snapshot) if validation.accepted else None, validation)


@dataclass
class RuntimeCheck:
    accepted: bool
    reason: str = ''
    nominal_envelope: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    actual_brake_envelope: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    position_error: float | None = None
    velocity_error: float | None = None

    @property
    def envelope(self):
        """Concatenated audit points; the two polylines are checked separately."""
        return np.vstack([self.nominal_envelope, self.actual_brake_envelope])


def observation_age(now, wall_time, source_time, captured):
    """Use the older of simulation and wall clocks; reject clock reversal."""
    values = np.asarray([now, wall_time, source_time, captured], dtype=float)
    if not np.isfinite(values).all() or now < source_time-1e-9 or wall_time < captured-1e-9:
        raise ValueError('Invalid or reversed observation clock')
    return max(0., float(now-source_time), float(wall_time-captured))


def _check_envelopes(safety, nominal, actual_brake, actual, position_error=None, velocity_error=None):
    result = RuntimeCheck(False, nominal_envelope=nominal, actual_brake_envelope=actual_brake,
                          position_error=position_error, velocity_error=velocity_error)
    # Joining these two independent polylines would invent a segment from the
    # nominal stopping endpoint back to the actual vehicle position.
    for name, points in (('REFERENCE', nominal), ('ACTUAL', actual_brake)):
        if not safety.clear(points):
            result.reason = name+'_OBSTACLE'
            return result
        if not safety.known(points, actual, body_padding=BODY_NUMERIC_TOLERANCE):
            result.reason = name+'_UNOBSERVED'
            return result
    result.accepted = True
    return result


def execution_check(command, elapsed, safety, *, actual_position, actual_velocity,
                    actual_acceleration, horizon, position_tolerance, velocity_tolerance,
                    reference_flight=None):
    """Recheck a short future prefix, its complete stop, and an actual-state stop.

    Call with fresh observation evidence. This geometric check alone does not
    renew observation freshness: the caller also checks ``observation_age`` and
    expires the execution lease no later than ``now + horizon``. Actual
    acceleration must be a measured/estimated value, not silently relabelled
    reference acceleration. The check is not a robust tracking-error proof.
    """
    elapsed = _nonnegative(elapsed, 'elapsed')
    horizon = _nonnegative(horizon, 'horizon')
    position_tolerance = _nonnegative(position_tolerance, 'position tolerance')
    velocity_tolerance = _nonnegative(velocity_tolerance, 'velocity tolerance')
    actual = _xyz(actual_position, 'actual position')
    actual_v = _xyz(actual_velocity, 'actual velocity')
    actual_a = _xyz(actual_acceleration, 'actual acceleration')
    p, v, _ = command.sample(elapsed)
    errors = dict(position_error=float(np.linalg.norm(actual-p)),
                  velocity_error=float(np.linalg.norm(actual_v-v)))
    if errors['position_error'] > position_tolerance:
        return RuntimeCheck(False, 'TRACKING_POSITION', **errors)
    if errors['velocity_error'] > velocity_tolerance:
        return RuntimeCheck(False, 'TRACKING_VELOCITY', **errors)
    if safety is None:
        return RuntimeCheck(False, 'NO_OBSERVATION', **errors)
    planning = reference_flight if reference_flight is not None else safety.flight
    if (not 0 < planning.max_acceleration <= safety.flight.max_acceleration
            or not 0 < planning.max_jerk <= safety.flight.max_jerk):
        raise ValueError('Reference stopping budget must not exceed physical limits')
    try:
        p, v, a = command.sample(elapsed+horizon)
        stop = QuinticPath.brake(p, v, a, planning.max_acceleration, planning.max_jerk)
        nominal = np.vstack([command.remaining(elapsed, actual, horizon), stop.remaining(0., p)])
        actual_stop = QuinticPath.brake(actual, actual_v, actual_a,
                                      safety.flight.max_acceleration, safety.flight.max_jerk)
        actual_envelope = actual_stop.remaining(0., actual)
    except ValueError:
        return RuntimeCheck(False, 'BRAKE_DYNAMICS', **errors)
    return _check_envelopes(safety, nominal, actual_envelope, actual, **errors)


def activation_check(candidate, command, command_start, now, wall_time, generation,
                     revision, safety, *, actual_position, actual_velocity, actual_acceleration,
                     deadline, max_observation_age, horizon, position_tolerance,
                     velocity_tolerance, clock_tolerance=1e-7):
    """Activate only at the promised tick; WAIT retains the current command.

    On success use ``command=candidate.path`` and
    ``command_start=candidate.request.switch_time``. A late result is discarded,
    never shifted to ``now`` or played from time zero after the cutover.
    """
    deadline = _nonnegative(deadline, 'deadline')
    max_observation_age = _nonnegative(max_observation_age, 'max observation age')
    clock_tolerance = _nonnegative(clock_tolerance, 'clock tolerance')
    request = candidate.request
    if generation != request.generation or revision != request.revision:
        return RuntimeCheck(False, 'SUPERSEDED')
    if not candidate.validation.accepted or candidate.path is None:
        return RuntimeCheck(False, 'CANDIDATE_'+candidate.validation.reason)
    age = observation_age(now, wall_time, request.source_time, request.captured)
    if age > min(deadline, max_observation_age):
        return RuntimeCheck(False, 'STALE')
    if now < request.switch_time-clock_tolerance:
        return RuntimeCheck(False, 'WAIT')
    if now > request.switch_time+clock_tolerance:
        return RuntimeCheck(False, 'MISSED_SWITCH')
    old_state = command.sample(request.switch_time-command_start)
    if (not np.allclose(old_state, request.start_pva, atol=1e-8, rtol=0.)
            or not np.allclose(candidate.path.sample(0.), request.start_pva, atol=1e-8, rtol=0.)):
        return RuntimeCheck(False, 'START_STATE')
    return execution_check(candidate.path, max(0., now-request.switch_time), safety,
        actual_position=actual_position, actual_velocity=actual_velocity,
        actual_acceleration=actual_acceleration, horizon=horizon,
        position_tolerance=position_tolerance, velocity_tolerance=velocity_tolerance)


def brake_from_reference(command, elapsed, safety, *, actual_position, actual_velocity,
                         actual_acceleration, flight=None, reference_flight=None):
    """Build one continuous reference brake; independently certify actual stop.

    The caller latches this brake and increments generation once. Subsequent
    failed/old results must not recreate it or restart its execution clock.
    An uncertified brake is returned with ``accepted=False`` for an honest log;
    this is a fallback command, not permission to claim certified execution.
    """
    elapsed = _nonnegative(elapsed, 'elapsed')
    actual = _xyz(actual_position, 'actual position')
    actual_v = _xyz(actual_velocity, 'actual velocity')
    actual_a = _xyz(actual_acceleration, 'actual acceleration')
    flight = safety.flight if safety is not None else flight
    if flight is None:
        raise ValueError('Braking requires explicit flight limits')
    planning = reference_flight if reference_flight is not None else flight
    if (not 0 < planning.max_acceleration <= flight.max_acceleration
            or not 0 < planning.max_jerk <= flight.max_jerk):
        raise ValueError('Reference braking budget must not exceed physical limits')
    p, v, a = command.sample(elapsed)
    brake = QuinticPath.brake(p, v, a, planning.max_acceleration, planning.max_jerk)
    nominal = brake.remaining(0., actual)
    if safety is None:
        return brake, RuntimeCheck(False, 'NO_OBSERVATION', nominal_envelope=nominal)
    try:
        actual_stop = QuinticPath.brake(actual, actual_v, actual_a,
                                      flight.max_acceleration, flight.max_jerk)
    except ValueError:
        return brake, RuntimeCheck(False, 'ACTUAL_BRAKE_DYNAMICS', nominal_envelope=nominal)
    return brake, _check_envelopes(safety, nominal, actual_stop.remaining(0., actual), actual)
