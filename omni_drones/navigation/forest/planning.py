"""单个异步 Qwen 请求与持续轨迹执行；请求节拍使用墙钟。"""
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from .geometry import (INFLATION, VERTICAL_RADIUS, BOUNDS, Z_BOUNDS,
    PROTOCOL_VERSION, occupied_centers, observed_centers, observed_body_height,
    local_to_world, world_to_local, parse_path, yaw_from_quat, inside_polygon)
from .geometry import center_clearance, clearance_margin, collision_free
from .observation import render_observation, representation_spec
from .trajectory import QuinticPath, connect_path


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def smooth_path(points, flight, check, start_vel=None, start_acc=None):
    for scale in (1., .5, .25):
        try:
            path = QuinticPath(points, flight.max_speed, flight.max_acceleration,
                start_vel=start_vel, start_acc=start_acc, max_jerk=flight.max_jerk,
                max_yaw_rate=flight.max_yaw_rate, max_yaw_acceleration=flight.max_yaw_acceleration,
                tangent_scale=scale, corner_distance=flight.corner_distance,
                max_path_deviation=flight.max_path_deviation, check=check)
        except ValueError:
            continue
        if check(path.remaining(0., points[0])):
            return path
    raise ValueError('no collision-free continuous connection')


def infer(model, snapshot, view):
    # 渲染、写盘和推理都在工作线程，主线程继续物理步进。
    from .qwen import task_prompt
    snap, folder = snapshot, snapshot['folder']
    folder.mkdir()
    np.savez_compressed(folder/'observation.npz', **{key: snap[key] for key in
        ('points', 'evidence', 'position', 'yaw', 'timestamp', 'goal', 'velocity', 'acceleration')},
        representation_json=json.dumps(representation_spec()), protocol_version=PROTOCOL_VERSION,
        flight_json=json.dumps(snap['flight']), lidar_hz=snap['lidar_hz'],
        corridor=snap['corridor'] if snap.get('corridor') is not None else np.empty((0,2)))
    render_observation(snap['points'], snap['evidence'], snap['goal'], folder/'input.png', view,
                       corridor=snap.get('corridor'), max_acceleration=snap['flight']['max_acceleration'])
    prompt = task_prompt(snap['goal'], view, snap.get('corridor'),
                         snap['velocity'], snap['acceleration'], PROTOCOL_VERSION)
    write_json(folder/'prompt.json', {'prompt': prompt})
    raw, latency = model.generate(folder/'input.png', prompt)
    result = dict(raw=raw, latency_seconds=latency,
                  pipeline_seconds=time.perf_counter()-snap['wall'])
    write_json(folder/'prediction.json', result)
    return result


class HeightBand:
    """World-Z band above the ground on a regular XY grid (picklable for the spline worker).

    Outside the grid, or where a limit is not finite, there is no bound.
    """
    def __init__(self, xs, ys, low, high):
        self.xs, self.ys = np.asarray(xs, float), np.asarray(ys, float)
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)
        if self.low.shape != (len(self.xs), len(self.ys)) or self.high.shape != self.low.shape:
            raise ValueError('band limits must be len(xs) x len(ys)')

    def limits(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        u = (xy[:, 0]-self.xs[0])/(self.xs[1]-self.xs[0])
        v = (xy[:, 1]-self.ys[0])/(self.ys[1]-self.ys[0])
        inside = (u >= 0) & (v >= 0) & (u <= len(self.xs)-1) & (v <= len(self.ys)-1)
        i = np.clip(np.floor(u).astype(int), 0, len(self.xs)-2)
        j = np.clip(np.floor(v).astype(int), 0, len(self.ys)-2)
        fu, fv = np.clip(u-i, 0., 1.), np.clip(v-j, 0., 1.)

        def blend(grid):
            value = ((1-fu)*(1-fv)*grid[i, j]+fu*(1-fv)*grid[i+1, j]
                     + (1-fu)*fv*grid[i, j+1]+fu*fv*grid[i+1, j+1])
            return np.where(inside & np.isfinite(value), value, np.nan)
        low, high = blend(self.low), blend(self.high)
        return np.where(np.isnan(low), -np.inf, low), np.where(np.isnan(high), np.inf, high)

    def leaving(self, points, tolerance=1e-3):
        """Ordered points stay in the band, or a leading run outside only moves back towards it."""
        points = np.asarray(points, float).reshape(-1, 3)
        low, high = self.limits(points[:, :2])
        excess = np.maximum(points[:, 2]-high, low-points[:, 2])
        outside = excess > 0.
        if not outside.any():
            return True
        lead = int(np.argmax(~outside)) if (~outside).any() else len(points)
        return bool(not outside[lead:].any() and (np.diff(excess[:lead]) <= tolerance).all())


class PathSafety:
    """教师与执行端共用的障碍、通道及已观测制动空间检查。"""
    height_bounds = None
    clearance_reserve = 0.

    def with_clearance_reserve(self, reserve):
        from copy import copy
        if not np.isfinite(reserve) or reserve < 0:
            raise ValueError('clearance reserve must be finite and nonnegative')
        view = copy(self)
        view.clearance_reserve = self.clearance_reserve+float(reserve)
        return view

    def __init__(self, points, evidence, corridor, flight, lidar_hz=10., height_bounds=None):
        self.obstacles = cKDTree(np.asarray(points).reshape(-1, 3))
        self.evidence = cKDTree(evidence)
        self.corridor, self.flight, self.lidar_hz = corridor, flight, lidar_hz
        self.height_bounds = height_bounds
        self.frame_position, self.frame_yaw = np.zeros(3), 0.

    def clear_points(self, centers, padding=0.):
        free = collision_free(self.obstacles, centers, padding+self.clearance_reserve,
                              center_clearance(self.flight))
        return free & self._admissible(centers)

    def _admissible(self, centers):
        """Window, corridor and height bounds; obstacles are checked separately."""
        local = world_to_local(centers, self.frame_position, self.frame_yaw)
        free = ((local >= [BOUNDS[0], BOUNDS[2], Z_BOUNDS[0]]) &
                (local <= [BOUNDS[1], BOUNDS[3], Z_BOUNDS[1]])).all(axis=1)
        if self.corridor is not None:
            free &= inside_polygon(np.asarray(centers)[:, :2], self.corridor)
        if self.height_bounds is not None:
            free &= (np.asarray(centers)[:, 2] >= self.height_bounds[0]) & (np.asarray(centers)[:, 2] <= self.height_bounds[1])
        return free

    def clear_leaving(self, centers, padding, start, tolerance=2e-3, require_exit=True):
        """``clear_points`` for an ordered curve that may begin inside the padding.

        Tracking error can leave a stopped body a centimetre inside the planning
        reserve; every curve from there then failed at its first sample and the
        vehicle never moved again (runs ed81189a, aa7fffd3: 0.86 m from a stem,
        0.87 m required). A leading run that fails only the reserve and padding
        is accepted while it stays outside the hard body envelope and never gets
        closer to an obstacle than ``start`` or any earlier sample. After that
        run the full padded check applies. A whole curve must also end outside
        the padding (``require_exit``); a connector or a brake envelope is only
        part of one and may end where it began.
        """
        centers = np.asarray(centers, float).reshape(-1, 3)
        return self._leaves(centers, self.clear_points(centers, padding), start, padding, tolerance, require_exit)

    def _leaves(self, centers, free, start, padding, tolerance=2e-3, require_exit=True, cap=None):
        """``clear_leaving`` given the padded check ``free`` of ``centers``.

        Leading samples are admissible but fail the padded check, so their
        margins are below padding+reserve; a cap above that changes no
        comparison. At the old 2 m cap one stopped spline validation spent
        0.65 s of 1.0 s in the neighbour query, every candidate was older than
        0.8 s when ready, and the vehicle never moved again (run 2f3009e0).
        """
        if free.all():
            return True
        lead = int(np.argmax(free)) if free.any() else len(free)
        if ((require_exit and lead == len(free)) or not free[lead:].all()
                or not self._admissible(centers[:lead]).all()):
            return False
        if cap is None:
            cap = float(np.max(padding))+self.clearance_reserve+.05
        margin = clearance_margin(self.obstacles, np.vstack([start, centers[:lead]]),
                                  center_clearance(self.flight), cap)
        floor = np.maximum.accumulate(margin)[:-1]-tolerance
        return bool((margin[1:] >= 0.).all() and (margin[1:] >= floor).all())

    def clear(self, path, start=None):
        """Sampled segment check; ``start`` admits a leading escape (``clear_leaving``)
        without requiring the segment itself to leave the padding."""
        path = np.asarray(path).reshape(-1, 3)
        reference = center_clearance(self.flight) is not None
        spacing = .04 if reference else .08
        samples = (np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/spacing))+1))
                                  for a, b in zip(path[:-1], path[1:])]) if len(path) > 1 else path)
        padding = .5*np.linalg.norm(np.diff(samples, axis=0), axis=1).max(initial=0.) if reference else .05
        if start is not None:
            return self.clear_leaving(samples, padding, start, require_exit=False)
        return bool(self.clear_points(samples, padding=padding).all())

    def clear_segments(self, start, ends, leaving=False):
        """``clear([start, end])`` for every end, in one obstacle query.

        Each call of ``clear`` rebuilds a neighbourhood tree; a stopped vehicle
        tried ~50 short steps one by one, 0.22 s per observation against 38k
        points, and every observation went stale (run 8ba92099). ``leaving``
        gives ``clear([start, end], start=start)``: a vehicle parked a few
        centimetres inside the padding otherwise failed every step at its own
        position, even straight away from the stem (run 393cb1bd).
        """
        start = np.asarray(start, float)
        ends = np.asarray(ends, float).reshape(-1, 3)
        if not len(ends):
            return np.zeros(0, bool)
        reference = center_clearance(self.flight) is not None
        spacing = .04 if reference else .08
        lengths = np.linalg.norm(ends-start, axis=1)
        counts = np.maximum(2, np.ceil(lengths/spacing).astype(int)+1)
        owner = np.repeat(np.arange(len(ends)), counts)
        fraction = np.concatenate([np.linspace(0., 1., n) for n in counts])
        samples = start+fraction[:, None]*(ends-start)[owner]
        padding = (.5*lengths/(counts-1))[owner] if reference else .05
        free = self.clear_points(samples, padding=padding)
        clear = np.bincount(owner, weights=(~free).astype(float), minlength=len(ends)) == 0
        if leaving:
            first = np.r_[0, np.cumsum(counts)]
            for i in np.flatnonzero(~clear):
                a, b = first[i], first[i+1]
                clear[i] = self._leaves(samples[a:b], free[a:b], start,
                                        padding[a:b] if reference else padding, require_exit=False)
        return clear

    def known(self, path, actual, body_padding=.05):
        return bool(self.known_points(path, actual, body_padding).all())

    def known_points(self, centers, actual, body_padding=.05):
        return observed_centers(self.evidence, centers, actual,
                                vertical=observed_body_height(self.flight.max_acceleration),
                                body_padding=body_padding)

    def braking_corridor(self, path, elapsed, actual):
        # 在下一轮安全检查之前继续一小段，仍须保有完整制动空间。
        horizon = 2./self.lidar_hz
        prefix = path.remaining(elapsed, actual, horizon)
        p, v, a = path.sample(elapsed+horizon)
        brake = QuinticPath.brake(p, v, a, self.flight.max_acceleration, self.flight.max_jerk)
        return np.vstack([prefix, brake.remaining(0., p)])

    def guard(self, path, elapsed, actual, body_padding=.05):
        corridor = self.braking_corridor(path, elapsed, actual)
        return self.clear(corridor) and self.known(corridor, actual, body_padding)

    def checks(self, points, actual):
        """Keep obstacle rejection separate from missing observation evidence."""
        return dict(obstacle_clear=self.clear(points), coverage_known=self.known(points, actual))

    def reconstruct(self, points, velocity, acceleration, actual):
        return smooth_path(points, self.flight,
                           lambda q: self.clear(q) and self.known(q, actual), velocity, acceleration)

    def observation_descent(self, start, actual, distance):
        """仅在新增机体空间有射线证据时，生成一次短距离下降观察。"""
        if distance <= 0:
            return None
        if distance > .2:
            raise ValueError('hold_descent must be between 0 and 0.2 m')
        points = np.linspace(start, np.asarray(start)-[0., 0., distance], 21)
        def safe(q):
            # 当前机体的额外 5 cm 余量不能作为探索未知空间的依据。
            return self.clear(q) and self.known(q, actual, body_padding=0.)
        if not safe(points):
            return None
        try:
            path = smooth_path(points[[0, -1]], self.flight, safe)
        except ValueError:
            return None
        if all(self.guard(path, t, actual, body_padding=0.)
               for t in np.linspace(0., path.duration, max(2, int(np.ceil(path.duration*self.lidar_hz))+1))):
            return path
        return None


class RollingPlanner(PathSafety):
    def __init__(self, cfg, model, output, start):
        self.cfg, self.flight, self.output = cfg, cfg.task.flight, output
        self.model = model
        self.lidar_hz = cfg.task.lidar.hz
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future, self.snapshot, self.observation = None, None, None
        self.corridor = None
        self.path, self.path_start = None, 0.
        self.hover = np.asarray(start, float)
        self.braking, self.terminal = False, None
        self.hold_timestamp, self.descent_used = None, False
        self.period = 1./cfg.planning.hz
        self.next_request = None
        self.started = time.perf_counter()
        self.events, self.completed, self.accepted = [], [], []
        self.requests, self.missed, self.overdue = 0, 0, 0
        self.decisions = Counter()

    def needs_snapshot(self, t):
        return (t >= self.cfg.history_seconds and not self.braking
                and (self.future is None or self.future.done()) and self.terminal is None
                and (self.next_request is None or time.perf_counter() >= self.next_request))

    def observe(self, snapshot):
        self.frame_position, self.frame_yaw = snapshot['position'], snapshot['yaw']
        world = local_to_world(snapshot['points'], snapshot['position'], snapshot['yaw'])
        self.obstacles = cKDTree(world)
        corridor=snapshot.get('corridor')
        self.corridor=(local_to_world(np.c_[corridor,np.zeros(len(corridor))],snapshot['position'],snapshot['yaw'])[:,:2]
                       if corridor is not None else None)
        if snapshot['evidence'] is None:
            return
        self.observation = dict(snapshot, flight=dict(self.flight), lidar_hz=self.lidar_hz)
        seen = local_to_world(snapshot['evidence'],
                              snapshot['position'], snapshot['yaw'])
        self.evidence = cKDTree(seen)

    def reference(self, t):
        if self.path is None:
            return self.hover.copy(), np.zeros(3), np.zeros(3)
        return self.path.sample(t-self.path_start)

    def stop(self, t, cause, terminal=False):
        if terminal:
            self.terminal = cause
        if self.braking:
            return
        if self.path is None:
            return
        p, v, a = self.reference(t)
        self.path = QuinticPath.brake(p, v, a, self.flight.max_acceleration, self.flight.max_jerk)
        self.path_start, self.braking = t, True
        self.events.append(dict(timestamp=t, reason=cause, action='brake'))

    def tick(self, t, state, scan_updated=False):
        now = time.perf_counter()
        if self.future is not None and self.future.done():
            result = self.future.result()
            snap = self.snapshot
            self.future = None
            self.completed.append(now-self.started)
            age = max(t-snap['timestamp'], now-snap['wall'])
            result.update(simulation_age=t-snap['timestamp'], wall_age=now-snap['wall'])
            self.overdue += int(now-snap['wall'] > self.period)
            decision = dict(timestamp=t, reason='accepted')
            try:
                if age > self.cfg.planning.max_snapshot_age:
                    raise ValueError('stale_observation')
                parsed = parse_path(result['raw'])
                if parsed['status'] == 'HOLD':
                    self.stop(t, 'model_hold')
                    decision['reason'] = 'model_hold'
                    if self.hold_timestamp is None:
                        self.hold_timestamp = t
                    # 停稳后的新观测仍得到 HOLD，才考虑一次下降观察。
                    if (self.terminal is None and self.path is None and not self.descent_used
                            and self.cfg.planning.get('hold_descent', 0.) > 0
                            and snap['timestamp'] > self.hold_timestamp
                            and np.linalg.norm(state[7:10]) < self.flight.settle_speed
                            and np.linalg.norm(state[:3]-self.hover) < self.flight.settle_error):
                        path = self.observation_descent(self.hover, state[:3],
                                                       self.cfg.planning.get('hold_descent', 0.))
                        if path is not None:
                            self.path, self.path_start, self.descent_used = path, t, True
                            decision['reason'] = 'observation_descent'
                            self.events.append(dict(timestamp=t, reason='model_hold', action='observe_down',
                                                    distance=float(self.hover[2]-path.points[-1, 2])))
                elif self.braking:
                    decision['reason'] = 'braking'
                elif self.hold_timestamp is not None and snap['timestamp'] <= self.hold_timestamp:
                    decision['reason'] = 'waiting_for_stopped_observation'
                elif self.terminal is None:
                    world = local_to_world(parsed['points'], snap['position'], snap['yaw'])
                    p, v, a = self.reference(t)
                    points = connect_path(np.vstack([snap['position'], world]), p, v, self.flight.max_acceleration)
                    path = self.reconstruct(points, v, a, state[:3])
                    direction = points[1]-points[0]
                    angle = np.arctan2(direction[1], direction[0])-yaw_from_quat(state[3:7])
                    angle = abs(np.arctan2(np.sin(angle), np.cos(angle)))
                    if np.linalg.norm(direction[:2]) > .02 and np.linalg.norm(v) < 1e-6 and np.linalg.norm(a) < 1e-6 and angle > .3:
                        path.hold_start(angle/self.flight.max_yaw_rate+
                                        self.flight.max_yaw_rate/self.flight.max_yaw_acceleration)
                    if not self.guard(path, 0., state[:3]):
                        raise ValueError('insufficient_observed_braking_space')
                    self.path, self.path_start, self.braking = path, t, False
                    self.hold_timestamp, self.descent_used = None, False
                    self.accepted.append(now-self.started)
                    decision.update(world_points=world.tolist(), connection_points=points.tolist(),
                                    start_velocity=v.tolist(), start_acceleration=a.tolist())
                else:
                    decision['reason'] = 'stopping'
            except ValueError as error:
                decision['reason'] = str(error)
            self.decisions[decision['reason']] += 1
            write_json(snap['folder']/'prediction.json', result)
            write_json(snap['folder']/'decision.json', decision)
            print(f'Plan {snap["folder"].name}: {decision["reason"]} '
                  f'inference={result["latency_seconds"]:.2f}s age={age:.2f}s '
                  f'accepted={len(self.accepted)}/{len(self.completed)}', flush=True)

        if scan_updated and self.path is not None and not self.braking:
            if not self.guard(self.path, t-self.path_start, state[:3]):
                self.stop(t, 'insufficient_observed_braking_space')
        if self.observation is not None and t-self.observation['timestamp'] > self.cfg.planning.max_snapshot_age:
            self.stop(t, 'stale_observation')

        if self.path is not None and t >= self.path_start+self.path.duration:
            self.hover = self.path.points[-1].copy()
            if self.braking and self.hold_timestamp is not None:
                self.hold_timestamp = t
            if not self.braking:
                self.events.append(dict(timestamp=t, reason='path_exhausted', action='hover'))
            self.path, self.braking = None, False

        ready = (scan_updated and self.observation is not None and t >= self.cfg.history_seconds
                 and not self.braking and self.terminal is None)
        if ready:
            if self.next_request is None:
                self.next_request = now
            if now >= self.next_request:
                slots = int((now-self.next_request)/self.period)+1
                self.next_request += slots*self.period
                self.missed += slots-int(self.future is None)
                if self.future is None:
                    self.snapshot = dict(self.observation, folder=self.output/f'plan_{self.requests:03d}',
                                         wall=self.observation.get('captured_wall', now))
                    self.future = self.pool.submit(infer, self.model, self.snapshot, self.cfg.view)
                    self.requests += 1
        return self.reference(t)

    def metrics(self):
        duration = time.perf_counter()-self.started
        intervals = np.diff(self.completed)
        return dict(requested_hz=self.cfg.planning.hz, requests=self.requests,
            completed=len(self.completed), accepted=len(self.accepted), missed_request_slots=self.missed,
            decision_counts=dict(self.decisions),
            overdue_results=self.overdue, output_intervals_seconds=intervals.tolist(),
            output_hz=(float(1/intervals.mean()) if len(intervals) else None),
            accepted_intervals_seconds=np.diff(self.accepted).tolist(),
            accepted_per_wall_second=len(self.accepted)/max(duration, 1e-9), events=self.events)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
