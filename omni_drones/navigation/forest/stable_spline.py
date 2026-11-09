"""One asynchronous spline candidate with a fixed future reference boundary.

This coordinator never changes the active command, its clock, or stop state. It
only returns a numerically compatible candidate at its promised controller tick.
The caller must separately certify it against its latest fresh observation and
the measured state before activation. Candidate age is not map freshness.
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import asdict, replace
from multiprocessing import get_context
import time

import numpy as np

from .observed_esdf import ObservedESDF
from .spline_flight import anchored_route
from .spline_optimizer import PythonSplineOptimizer
from .spline_runtime import ExecutableSpline, SplineCandidate, make_handoff, observation_age
from .local import BODY_NUMERIC_TOLERANCE


def height_guided_route(route, target, actual, safety):
    """Recover a small public height slope lost by the coarse A* z grid.

    Keep actual vertical obstacle maneuvers and the exact future anchor. A
    distance-shortened route gets the corresponding fraction of target height;
    it does not have to reach the farther target to start climbing. Only an
    independently body-observed, clear correction may replace the coarse route.
    """
    route = np.asarray(route, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(route) < 2 or np.ptp(route[:, 2]) > .05:
        return route
    delta = target-route[0]
    span = float(delta[:2]@delta[:2])
    if span < .04 or abs(delta[2]) < BODY_NUMERIC_TOLERANCE:
        return route
    progress = np.clip((route[:, :2]-route[0, :2])@delta[:2]/span, 0., 1.)
    for fraction in (1., .5, .25):
        proposed = route.copy()
        proposed[:, 2] = route[0, 2]+fraction*progress*delta[2]
        dense = np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/.04))+1))
                                for a, b in zip(proposed[:-1], proposed[1:])])
        if safety.clear(dense) and safety.known(dense, actual, body_padding=BODY_NUMERIC_TOLERANCE):
            return proposed
    return route


def level_route(route):
    """The same route at its start height, for a descent into unobserved space.

    The upright MID-360 (lowest ray -7 deg) and the downward cone (to -30 deg)
    leave a band below body height unobserved 1-5 m ahead. From a stop, a 0.4 m
    A* step down there failed every curve and brake check for 20 s (run
    4b639dae), while the route at the current height was observed. Flying it
    brings the lower space into view; later routes then descend.
    """
    level = np.asarray(route, dtype=float).copy()
    level[:, 2] = level[0, 2]
    return level


class StableSplinePlanner:
    """Bounded single-worker optimization, immutable handoff and stale rejection."""

    def __init__(self, planning_flight, activation_lead=.4, candidate_max_age=.8, control_dt=.01,
                 process_worker=False, profile_acceleration=None):
        values = np.asarray([activation_lead, candidate_max_age, control_dt], dtype=float)
        if (not np.isfinite(values).all() or (values <= 0).any()
                or candidate_max_age < activation_lead):
            raise ValueError('Positive finite timing and candidate age >= activation lead are required')
        self.activation_lead = float(activation_lead)
        self.candidate_max_age = float(candidate_max_age)
        self.control_dt = float(control_dt)
        self.clock_tolerance = max(1e-7, self.control_dt*1e-5)
        # Retiming must rebuild exact moving p/v/a constraints each time. The
        # original three attempts can remain just above a tighter planning
        # budget even when a slower feasible reference exists.
        self.optimizer = PythonSplineOptimizer(planning_flight, retimings=6,
                                               enforce_feasible_controls=True,
                                               profile_acceleration=profile_acceleration)
        self.process_worker = bool(process_worker)
        self._pool = (ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'))
                      if self.process_worker else ThreadPoolExecutor(max_workers=1, thread_name_prefix='stable-spline'))
        self._future = self._request = self._ready = None
        self._ready_reported = False
        self._closed = False
        self._last_poll = None

    def __getstate__(self):
        # Spawn only the pure CPU optimizer, never the executor, simulator,
        # in-flight futures or CUDA state. _optimize uses this member alone.
        return dict(optimizer=self.optimizer)

    @property
    def busy(self):
        return self._future is not None or self._ready is not None

    @property
    def ready(self):
        return self._ready is not None and self._ready['candidate'] is not None

    @property
    def can_submit(self):
        return not self._closed and not self.busy

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Spline planner is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def close(self):
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._future = self._request = self._ready = None

    def submit(self, command, command_start, now, snapshot, generation, revision, target, speed):
        """Freeze future p/v/a; return False while another candidate is pending."""
        if self._closed:
            raise RuntimeError('Spline planner is closed')
        if self.busy:
            return False
        target = np.asarray(target, dtype=float)
        measured = np.asarray(snapshot['state'], dtype=float).copy()
        route = np.asarray(snapshot['decision']['route'], dtype=float).copy()
        if (target.shape != (3,) or not np.isfinite(target).all()
                or measured.ndim != 1 or len(measured) < 3 or not np.isfinite(measured).all()
                or route.ndim != 2 or route.shape[1] != 3 or not len(route) or not np.isfinite(route).all()
                or not np.isfinite(speed) or speed <= 0):
            raise ValueError('Spline request needs finite state/route/target and positive speed')
        request = make_handoff(command, command_start, now, snapshot['timestamp'], snapshot['captured'],
                               generation, revision, activation_lead=self.activation_lead,
                               control_dt=self.control_dt)
        # Observation snapshots and their safety/field objects are immutable to
        # workers. Copy request-local arrays so goal handoffs cannot alter them.
        captured = dict(snapshot, state=measured, decision=dict(snapshot['decision'], route=route))
        self._request = request
        self._ready_reported = False
        self._future = self._pool.submit(self._optimize, captured, request, target.copy(), float(speed))
        return True

    def _optimize(self, snapshot, request, target, speed):
        began = time.perf_counter()
        result = dict(candidate=None, optimization=None, field_reused=False)
        try:
            actual = snapshot['state'][:3]
            safety = snapshot.get('planning_safety', snapshot['safety'])
            route = anchored_route(snapshot['decision']['route'], request.start_pva[0])
            if len(route) < 2:
                result['optimization'] = dict(validation=dict(accepted=False, reason='NO_ROUTE'))
            else:
                field = snapshot.get('distance_field')
                result['field_reused'] = field is not None
                if field is None:
                    field = ObservedESDF(snapshot['points'], snapshot['evidence'],
                        bounds=np.array([actual-[5., 4., 2.], actual+[5., 4., 2.]]))
                guided = height_guided_route(route, target, actual, safety)
                routes = [('height_guided', guided), ('observed_astar', route)] if not np.array_equal(
                    guided, route) else [('observed_astar', route)]
                if route[-1, 2] < route[0, 2]-.05:
                    routes.append(('level_hold', level_route(route)))
                trials = []
                for route_name, proposed in routes:
                    if route_name == 'level_hold' and not all(
                            t['validation']['reason'].startswith('UNOBSERVED') for t in trials):
                        # Only an unobserved descent is held level; an obstacle
                        # answer of the lower routes stands.
                        break
                    if (route_name != 'level_hold' and 1e-8 < np.linalg.norm(proposed[-1]-target) < .6
                            and safety.clear([proposed[-1], target])
                            and safety.known([proposed[-1], target], actual)):
                        proposed = np.vstack([proposed, target])
                    optimized = self.optimizer.optimize(proposed, *request.start_pva, field, safety,
                                                        nominal_speed=speed, actual_position=actual)
                    trials.append(dict(route=route_name, validation=asdict(optimized.validation),
                                       attempts=optimized.attempts))
                    result['optimization'] = dict(validation=asdict(optimized.validation),
                                                 attempts=optimized.attempts, route_trials=trials)
                    if optimized.path is not None and optimized.validation.accepted:
                        # A clear height-guided polyline can still smooth into
                        # unknown voxels. Retry the original observed A* route;
                        # both routes retain identical full-curve/stop validation
                        # and the same immutable future p/v/a boundary.
                        result['candidate'] = SplineCandidate(request, ExecutableSpline(optimized.path),
                                                              optimized.validation)
                        break
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        result['seconds'] = time.perf_counter()-began
        return result

    def _event(self, reason, now, wall_time):
        request, result = self._request, self._ready
        return dict(event='spline_candidate', reason=reason, generation=request.generation,
                    revision=request.revision, switch_time=request.switch_time,
                    source_time=request.source_time,
                    candidate_age_seconds=observation_age(now, wall_time, request.source_time, request.captured),
                    compute_seconds=result['seconds'], optimization=result['optimization'],
                    field_reused=result['field_reused'], error=result.get('error'))

    def _discard(self):
        self._request = self._ready = None
        self._ready_reported = False

    def poll(self, now, wall_time, generation, revision, command, command_start):
        """Return a candidate only at the fixed tick, without activating it.

        A READY event certifies timing, version and exact reference continuity;
        it does not authorize geometry or actual-state execution. Rejections do
        not change the supplied command or command_start. WAIT is emitted once.
        """
        if self._closed:
            raise RuntimeError('Spline planner is closed')
        if (not np.isfinite([now, wall_time, command_start]).all()
                or (self._last_poll is not None and now < self._last_poll-1e-9)):
            raise ValueError('Polling requires finite clocks and monotonic simulation time')
        self._last_poll = float(now)
        if self._future is not None:
            if not self._future.done():
                return None, None
            try:
                self._ready = self._future.result()
                if self.process_worker and self._ready['candidate'] is not None:
                    candidate = self._ready['candidate']
                    self._ready['candidate'] = replace(candidate, path=ExecutableSpline(candidate.path.spline))
            except Exception as error:
                self._ready = dict(candidate=None, optimization=None, field_reused=False,
                                   seconds=None, error=f'Worker {type(error).__name__}: {error}')
            self._future = None
        if self._ready is None:
            return None, None
        request, result = self._request, self._ready
        age = observation_age(now, wall_time, request.source_time, request.captured)
        reason = None
        if generation != request.generation or revision != request.revision:
            reason = 'SUPERSEDED'
        elif age > self.candidate_max_age:
            reason = 'STALE_CANDIDATE'
        elif result['candidate'] is None:
            reason = 'NO_CANDIDATE'
        elif now > request.switch_time+self.clock_tolerance:
            reason = 'MISSED_SWITCH'
        elif now < request.switch_time-self.clock_tolerance:
            if not self._ready_reported:
                self._ready_reported = True
                return None, self._event('WAIT', now, wall_time)
            return None, None
        else:
            old_state = np.asarray(command.sample(request.switch_time-command_start))
            new_state = np.asarray(result['candidate'].path.sample(0.))
            if (not np.allclose(old_state, request.start_pva, atol=1e-8, rtol=0.)
                    or not np.allclose(new_state, request.start_pva, atol=1e-8, rtol=0.)):
                reason = 'START_STATE'
        event = self._event(reason or 'READY', now, wall_time)
        candidate = result['candidate'] if reason is None else None
        self._discard()
        return candidate, event
