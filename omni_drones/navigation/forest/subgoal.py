"""Observable execution contract and measured completion; no simulator imports."""
from dataclasses import asdict, dataclass
import math
import numpy as np


def angle_error(target, actual):
    return float(np.arctan2(np.sin(target-actual), np.cos(target-actual)))


@dataclass(frozen=True)
class Subgoal:
    goal_id: str
    revision: int
    target_world: tuple[float, float, float]
    arrival_yaw: float | None = None
    position_tolerance: float = .3
    speed_tolerance: float = .15
    yaw_tolerance: float = .15
    dwell_seconds: float = .5
    timeout_seconds: float = 40.
    no_motion_seconds: float = 12.
    completion_mode: str = 'stop'
    approach_direction: tuple[float, float, float] | None = None
    lateral_tolerance: float | None = None
    height_tolerance: float | None = None
    crossing_window_seconds: float = .25
    # Moving without getting closer (e.g. shuttling inside a dead-end pocket)
    # never trips ``no_motion``; this ends the goal so the mission can recover.
    no_progress_seconds: float | None = None
    no_progress_distance: float = .5

    def __post_init__(self):
        target = np.asarray(self.target_world, dtype=float)
        if not self.goal_id or not isinstance(self.revision, int) or self.revision < 1:
            raise ValueError('goal_id and a positive integer revision are required')
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('target_world must contain three finite coordinates')
        object.__setattr__(self, 'target_world', tuple(float(x) for x in target))
        for name in ('position_tolerance', 'speed_tolerance', 'yaw_tolerance',
                     'timeout_seconds', 'no_motion_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.arrival_yaw is not None and not math.isfinite(self.arrival_yaw):
            raise ValueError('arrival_yaw must be finite')
        if self.completion_mode not in ('stop', 'through'):
            raise ValueError('completion_mode must be stop or through')
        if (not math.isfinite(self.dwell_seconds) or self.dwell_seconds < 0
                or (self.completion_mode == 'stop' and not 0 < self.dwell_seconds < self.timeout_seconds)):
            raise ValueError('stop dwell must be positive and shorter than timeout; through dwell may be zero')
        for name in ('lateral_tolerance', 'height_tolerance'):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be positive when provided')
        if not math.isfinite(self.crossing_window_seconds) or self.crossing_window_seconds <= 0:
            raise ValueError('crossing_window_seconds must be finite and positive')
        if self.no_progress_seconds is not None and (not math.isfinite(self.no_progress_seconds)
                or self.no_progress_seconds <= 0):
            raise ValueError('no_progress_seconds must be finite and positive when provided')
        if not math.isfinite(self.no_progress_distance) or self.no_progress_distance <= 0:
            raise ValueError('no_progress_distance must be finite and positive')
        if self.completion_mode == 'through':
            direction = np.asarray(self.approach_direction, dtype=float).copy()
            if (direction.shape != (3,) or not np.isfinite(direction).all()
                    or abs(direction[2]) > 1e-9 or np.linalg.norm(direction[:2]) < 1e-9):
                raise ValueError('through goals require a finite horizontal approach_direction')
            direction /= np.linalg.norm(direction)
            object.__setattr__(self, 'approach_direction', tuple(float(x) for x in direction))
        elif self.approach_direction is not None:
            raise ValueError('approach_direction is only meaningful for through goals')

    def to_dict(self):
        return asdict(self)


class GoalMonitor:
    """Measured stop/dwell or directional passage through a bounded gate.

    A through goal needs consecutive actual-state samples straddling its plane
    in the forward direction. Both samples must lie within a short gate slab,
    lateral and height bounds, and the configured observation time window. A
    remote interpolation across the target, starting beyond it, or a reference
    trajectory cannot complete the gate. The final mission goal should be stop.
    """
    def __init__(self, goal, start_time=0.):
        self.goal, self.start_time = goal, float(start_time)
        self.inside_since = None
        self.last_time = self.start_time
        self.last_motion = self.start_time
        self.anchor = None
        self.anchor_yaw = None
        self.terminal = None
        self.previous_position = None
        self.previous_time = None
        self.crossing_position = None
        self.crossing_time = None
        self.progress_distance = None
        self.progress_time = self.start_time

    def update(self, now, position, velocity, yaw, failure=None):
        position, velocity = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
        if (position.shape != (3,) or velocity.shape != (3,)
                or not np.isfinite(np.r_[now, yaw, position, velocity]).all() or now < self.last_time):
            raise ValueError('state must be finite and time must be monotonic')
        self.last_time = now
        if (self.anchor is None or np.linalg.norm(position-self.anchor) >= .05
                or abs(angle_error(yaw, self.anchor_yaw)) >= .05):
            self.last_motion, self.anchor, self.anchor_yaw = now, position.copy(), yaw
        distance = float(np.linalg.norm(position-self.goal.target_world))
        if self.progress_distance is None or distance <= self.progress_distance-self.goal.no_progress_distance:
            self.progress_distance, self.progress_time = distance, now
        speed = float(np.linalg.norm(velocity))
        yaw_error = (0. if self.goal.arrival_yaw is None
                     else abs(angle_error(self.goal.arrival_yaw, yaw)))
        settled = (distance <= self.goal.position_tolerance and speed <= self.goal.speed_tolerance
                   and yaw_error <= self.goal.yaw_tolerance)
        self.inside_since = (now if self.inside_since is None else self.inside_since) if settled else None
        dwell = 0. if self.inside_since is None else now-self.inside_since
        through, gate = False, {}
        if self.goal.completion_mode == 'through':
            direction = np.asarray(self.goal.approach_direction)
            lateral_axis = np.array([-direction[1], direction[0], 0.])
            delta = position-self.goal.target_world
            signed, lateral, height = float(delta@direction), float(abs(delta@lateral_axis)), float(abs(delta[2]))
            lateral_limit = self.goal.lateral_tolerance or self.goal.position_tolerance
            height_limit = self.goal.height_tolerance or self.goal.position_tolerance
            gate = dict(gate_signed_distance_m=signed, gate_lateral_error_m=lateral,
                        gate_height_error_m=height)
            if self.previous_position is not None and now > self.previous_time:
                previous_delta = self.previous_position-self.goal.target_world
                previous_signed = float(previous_delta@direction)
                within_gate = (lateral <= lateral_limit and height <= height_limit
                    and abs(previous_delta@lateral_axis) <= lateral_limit
                    and abs(previous_delta[2]) <= height_limit)
                through = bool(within_gate and now-self.previous_time <= self.goal.crossing_window_seconds
                    and -self.goal.position_tolerance <= previous_signed < 0. <= signed <= self.goal.position_tolerance
                    and float(velocity@direction) > 1e-6)
                if through and self.crossing_time is None:
                    fraction = -previous_signed/(signed-previous_signed)
                    self.crossing_position = self.previous_position+fraction*(position-self.previous_position)
                    self.crossing_time = float(self.previous_time+fraction*(now-self.previous_time))
            self.inside_since, dwell = None, 0.
        self.previous_position, self.previous_time = position.copy(), float(now)
        if self.terminal is None:
            if failure:
                self.terminal = failure
            elif (through if self.goal.completion_mode == 'through' else dwell >= self.goal.dwell_seconds):
                self.terminal = 'goal_reached'
            elif now-self.start_time >= self.goal.timeout_seconds:
                self.terminal = 'timeout'
            elif now-self.last_motion >= self.goal.no_motion_seconds:
                self.terminal = 'no_motion'
            elif (self.goal.no_progress_seconds is not None and distance > 2*self.goal.position_tolerance
                    and now-self.progress_time >= self.goal.no_progress_seconds):
                self.terminal = 'no_progress'
        return dict(goal_id=self.goal.goal_id, revision=self.goal.revision,
                    status='COMPLETE' if self.terminal == 'goal_reached' else 'FAILED' if self.terminal else 'RUNNING',
                    reason=self.terminal or '', distance_m=distance, speed_mps=speed,
                    yaw_error_rad=yaw_error, dwell_seconds=dwell,
                    no_motion_seconds=now-self.last_motion, completion_mode=self.goal.completion_mode,
                    crossing_time=self.crossing_time,
                    crossing_position=(self.crossing_position.tolist() if self.crossing_position is not None else None),
                    **gate)


class HeadingReference:
    """Bounded yaw-rate and yaw-acceleration reference, unwrapped internally."""
    def __init__(self, yaw, max_rate, max_acceleration):
        if not np.isfinite([yaw, max_rate, max_acceleration]).all() or min(max_rate, max_acceleration) <= 0:
            raise ValueError('invalid heading limits')
        self.yaw, self.rate = float(yaw), 0.
        self.max_rate, self.max_acceleration = max_rate, max_acceleration

    def step(self, target, dt):
        if not np.isfinite([target, dt]).all() or not 0 < dt <= .1:
            raise ValueError('heading step requires finite target and 0 < dt <= .1')
        error = angle_error(target, self.yaw)
        acceleration = np.clip(4.*error-4.*self.rate, -self.max_acceleration, self.max_acceleration)
        rate = float(np.clip(self.rate+acceleration*dt, -self.max_rate, self.max_rate))
        acceleration = (rate-self.rate)/dt
        self.yaw += .5*(self.rate+rate)*dt
        self.rate = rate
        return self.yaw, self.rate, float(acceleration)
