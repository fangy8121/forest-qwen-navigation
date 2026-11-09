"""观测驱动的局部 A*、速度参考与短执行段/制动监督；不读取场景真值。"""
from collections import deque
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import label
from .esdf import astar
from .geometry import GRID_RESOLUTION, quat_matrix, local_to_world, world_to_local
from .observation import RAY_STEP
from .planning import PathSafety
from .trajectory import QuinticPath

BODY_NUMERIC_TOLERANCE = 1e-5  # 世界坐标 float32 的数值余量，不能用于厘米级探索。
# A fresh A* route ending this close (XY) to its target was limited by the lookahead.
ROUTE_TARGET_GAP = .6


class LocalMap:
    """Timed ray evidence, optionally in a fixed-size world-aligned moving window.

    The default retains the original route-bounded map. ``rolling=True`` keeps
    ``window_size`` metres of voxel centres around the most recent sensor pose;
    recentering copies only overlap and initializes every entering cell unknown.
    This is local observation memory, not a persistent SLAM or coverage map.

    ``memory_seconds`` (>= ``seconds``) keeps per-voxel free and hit evidence
    longer for a static scene. An upright MID-360 (-7..+52 deg) never sees the
    lower body volume near the vehicle; it is observed only from >2 m away, so a
    3 s window loses it (run 198e22c3). Remembered hits stay obstacles at their
    exact positions, so free space is never remembered around a forgotten hit.
    """
    def __init__(self, route, seconds=3., padding=6., ray_traversal='sampled_v1', *,
                 rolling=False, window_size=(24., 20., 10.), memory_seconds=None, evidence_range=6.):
        if ray_traversal not in ('sampled_v1', 'voxel_dda_v1'):
            raise ValueError('unknown free-evidence traversal')
        route = np.asarray(route, dtype=float)
        if (route.ndim != 2 or route.shape[1] != 3 or not len(route)
                or not np.isfinite(route).all()):
            raise ValueError('LocalMap route must be a nonempty array of finite world XYZ points')
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('LocalMap history must be finite and positive')
        # Free space is recorded along each ray up to this range. With an upright
        # MID-360 (-7 deg lowest ray) the body volume of a descent of dh is seen
        # only from >(0.26+dh)/tan(7 deg) away, beyond 6 m for dh=0.5 m (run 976f816b).
        if not np.isfinite(evidence_range) or evidence_range <= 0:
            raise ValueError('evidence_range must be finite and positive')
        self.evidence_range = float(evidence_range)
        memory_seconds = seconds if memory_seconds is None else memory_seconds
        if not np.isfinite(memory_seconds) or memory_seconds < seconds:
            raise ValueError('LocalMap memory must be finite and at least the raw-scan history')
        self.ray_traversal = ray_traversal
        self.rolling = bool(rolling)
        self.recenter_count = 0
        self._latest_timestamp = -np.inf
        if self.rolling:
            self.window_size = np.asarray(window_size, dtype=float)
            if (self.window_size.shape != (3,) or not np.isfinite(self.window_size).all()
                    or (self.window_size < 2*GRID_RESOLUTION).any()):
                raise ValueError('Rolling window_size must contain three finite metre spans >= two voxels')
            shape = np.ceil(self.window_size/GRID_RESOLUTION).astype(int)+1
            self._origin_cell = np.floor(route[0]/GRID_RESOLUTION).astype(np.int64)-shape//2
            self.origin = self._origin_cell*GRID_RESOLUTION
        else:
            self.window_size = None
            self.origin = np.floor((route.min(axis=0)-[padding, padding, 3.])/GRID_RESOLUTION)*GRID_RESOLUTION
            upper = route.max(axis=0)+[padding, padding, 3.]
            shape = np.ceil((upper-self.origin)/GRID_RESOLUTION).astype(int)+1
        self.seen = np.full(shape, -np.inf, dtype=np.float32)
        self.occupied = self.seen.copy()
        self.hit = np.full((*self.seen.shape, 3), np.nan)
        self.seconds, self.memory_seconds = seconds, float(memory_seconds)
        self.frames = deque()
        if ray_traversal == 'voxel_dda_v1':
            from .ray_evidence import trace_free_voxels
            trace_free_voxels(np.zeros((2, 2, 2), bool), np.zeros(3),
                             np.array([[1., 0., 0.]]), np.array([.1]), np.zeros(3), GRID_RESOLUTION)

    def _cells(self, points):
        cells = self._cell_indices(points)
        return cells[((cells >= 0) & (cells < self.seen.shape)).all(axis=1)]

    def _cell_indices(self, points):
        if self.rolling:
            # Quantize in the world lattice before subtracting the moving
            # origin. At exact half-voxel ties np.rint's even-cell rule must
            # not change merely because the window shifted by an odd cell.
            return np.rint(np.asarray(points)/GRID_RESOLUTION).astype(np.int64)-self._origin_cell
        return np.rint((points-self.origin)/GRID_RESOLUTION).astype(np.int32)

    def _inside_window(self, points):
        cells = self._cell_indices(points)
        return ((cells >= 0) & (cells < self.seen.shape)).all(axis=1)

    def recenter(self, position):
        """Translate the rolling window on the global voxel lattice, without sensing.

        No modulo wrap is used: cells leaving the window and their raw hits are
        forgotten. Returning later cannot recover evidence without a new scan.
        """
        if not self.rolling:
            return False
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Rolling map centre must be finite world XYZ')
        shape = np.asarray(self.seen.shape, dtype=np.int64)
        origin_cell = np.floor(position/GRID_RESOLUTION).astype(np.int64)-shape//2
        shift = origin_cell-self._origin_cell
        if not shift.any():
            return False
        seen, occupied = np.full_like(self.seen, -np.inf), np.full_like(self.occupied, -np.inf)
        hit = np.full_like(self.hit, np.nan)
        if (np.abs(shift) < shape).all():
            old_region = tuple(slice(max(0, int(d)), min(int(n), int(n+d))) for n, d in zip(shape, shift))
            new_region = tuple(slice(max(0, int(-d)), min(int(n), int(n-d))) for n, d in zip(shape, shift))
            seen[new_region], occupied[new_region] = self.seen[old_region], self.occupied[old_region]
            hit[new_region] = self.hit[old_region]
        self.seen, self.occupied, self.hit = seen, occupied, hit
        self._origin_cell, self.origin = origin_cell, origin_cell*GRID_RESOLUTION
        self.frames = deque((t, hits[self._inside_window(hits)]) for t, hits in self.frames)
        self.recenter_count += 1
        return True

    def add(self, frame):
        t = float(frame['timestamp'])
        if not np.isfinite(t):
            raise ValueError('Scan timestamp must be finite')
        if self.rolling and t < self._latest_timestamp-1e-8:
            raise ValueError('Rolling map scans must have monotonic timestamps')
        self.recenter(frame['sensor_position'])
        rotation = quat_matrix(frame['sensor_quaternion'])
        hits = frame['points_sensor'][frame['valid']] @ rotation.T+frame['sensor_position']
        if self.rolling:
            hits = hits[self._inside_window(hits)]
        self.frames.append((t, hits))
        while self.frames and self.frames[0][0] < t-self.seconds-1e-8:
            self.frames.popleft()
        distances = np.minimum(frame['distances'], float(frame['max_range']))
        if self.ray_traversal == 'voxel_dda_v1':
            from .ray_evidence import trace_free_voxels
            visited = np.zeros(self.seen.shape, dtype=bool)
            trace_free_voxels(visited, np.asarray(frame['sensor_position'], dtype=float),
                             np.asarray(frame['directions_world'], dtype=float),
                             np.minimum(distances, self.evidence_range).astype(float), self.origin, GRID_RESOLUTION)
            self.seen[visited] = t
        else:
            steps = np.arange(0., min(self.evidence_range, float(frame['max_range'])), RAY_STEP)
            for start in range(0, len(distances), 512):
                along = frame['sensor_position']+frame['directions_world'][start:start+512, None]*steps[None, :, None]
                cells = self._cells(along[steps[None, :] < distances[start:start+512, None]])
                self.seen[tuple(cells.T)] = t
        inside = ((self._cell_indices(hits) >= 0) & (self._cell_indices(hits) < self.seen.shape)).all(axis=1)
        cells = self._cell_indices(hits)[inside]
        self.occupied[tuple(cells.T)] = t
        self.hit[tuple(cells.T)] = hits[inside]
        self._latest_timestamp = max(t, self._latest_timestamp)

    def mark_observed_column(self, position, timestamp, depth, radius):
        """Mark the cylinder a body swept on a vertical take-off as observed free.

        The vehicle starts in the air at its take-off column; an upright LiDAR
        never sees the lower body volume there. Voxels with any recorded hit
        stay unobserved, and the marks age out like scan evidence.
        """
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all() or depth < 0 or radius <= 0:
            raise ValueError('Take-off column needs a finite position, depth >= 0 and radius > 0')
        r = np.arange(-radius, radius+1e-9, GRID_RESOLUTION/2)
        x, y, z = np.meshgrid(r, r, np.arange(-depth, GRID_RESOLUTION+1e-9, GRID_RESOLUTION/2), indexing='ij')
        offsets = np.c_[x.ravel(), y.ravel(), z.ravel()]
        offsets = offsets[np.linalg.norm(offsets[:, :2], axis=1) <= radius]
        cells = np.unique(self._cells(position+offsets), axis=0)
        cells = cells[~np.isfinite(self.occupied[tuple(cells.T)])]
        self.seen[tuple(cells.T)] = np.float32(timestamp)
        return len(cells)

    def snapshot(self, now, position):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(now):
            raise ValueError('Snapshot needs a finite timestamp and world XYZ position')
        extent = np.array([5., 4., 2.])
        lo = np.clip(np.floor((position-extent-self.origin)/GRID_RESOLUTION).astype(int), 0, self.seen.shape)
        hi = np.clip(np.ceil((position+extent-self.origin)/GRID_RESOLUTION).astype(int)+1, 0, self.seen.shape)
        if (hi <= lo).any():
            return np.empty((0, 3)), np.empty((0, 3))
        region = tuple(slice(a, b) for a, b in zip(lo, hi))
        # Later scans must never authorize a past free-space query. Historical
        # queries are conservative because newer stamps may overwrite old ones.
        memory = now-self.memory_seconds-1e-8
        active = ((self.seen[region] >= memory)
                  & (self.seen[region] <= np.float32(now))
                  & (self.occupied[region] < memory))
        evidence = self.origin+(np.argwhere(active)+lo)*GRID_RESOLUTION
        points = [p[(np.abs(p-position) <= extent+[1., 1., .5]).all(axis=1)]
                  for t, p in self.frames if now-self.seconds-1e-8 <= t <= now]
        if self.memory_seconds > self.seconds:
            # Hits older than the raw-scan window but inside the memory stay
            # obstacles, at the exact position of the last return in the voxel.
            old = ((self.occupied[region] >= memory) & (self.occupied[region] < now-self.seconds-1e-8))
            points.append(self.hit[region][old])
        return (np.concatenate(points) if points else np.empty((0, 3))), evidence


class VelocityTrajectory:
    """连续速度/加速度参考；积分位置仅用于执行段的几何检查。"""
    def __init__(self, position, velocity, acceleration, target, flight):
        self.position, self.target = np.asarray(position), np.asarray(target)
        delta = np.linalg.norm(self.target-velocity)
        duration = max(.25, 2*delta/flight.max_acceleration,
                       np.sqrt(6*delta/flight.max_jerk), 2*np.linalg.norm(acceleration)/flight.max_jerk)
        for _ in range(12):
            curve = CubicHermiteSpline([0., duration], [velocity, target], [acceleration, np.zeros(3)])
            times = np.linspace(0., duration, 81)
            if (np.linalg.norm(curve(times, 1), axis=1).max() <= flight.max_acceleration*1.001
                    and np.linalg.norm(curve(times, 2), axis=1).max() <= flight.max_jerk*1.001):
                break
            duration *= 1.2
        else:
            raise ValueError('velocity transition exceeds acceleration/jerk limits')
        self.curve, self.integral, self.duration = curve, curve.antiderivative(), duration

    def sample(self, t):
        t = np.asarray(t)
        clipped = np.clip(t, 0., self.duration)
        p = self.position+self.integral(clipped)-self.integral(0.)+np.maximum(0., t-self.duration)[..., None]*self.target
        v = np.where((t <= self.duration)[..., None], self.curve(clipped), self.target)
        a = np.where((t <= self.duration)[..., None], self.curve(clipped, 1), 0.)
        return p, v, a

    def remaining(self, t, actual, horizon=.2):
        times = np.linspace(t, t+horizon, max(2, int(np.ceil(horizon/.03))+1))
        return np.vstack([actual, self.sample(times)[0]])


def certificate(safety, path, position, horizon=.2):
    p, v, a = path.sample(horizon)
    brake = QuinticPath.brake(p, v, a, safety.flight.max_acceleration, safety.flight.max_jerk)
    envelope = np.vstack([path.remaining(0., position, horizon), brake.remaining(0., p)])
    _, velocities, accelerations = path.sample(np.linspace(0., horizon, 11))
    if (np.linalg.norm(velocities, axis=1).max() > safety.flight.max_speed*1.001
            or np.linalg.norm(accelerations, axis=1).max() > safety.flight.max_acceleration*1.001):
        return 'DYNAMICS', envelope
    if not safety.clear(envelope):
        return 'OBSTACLE', envelope
    if not safety.known(envelope, position, body_padding=BODY_NUMERIC_TOLERANCE):
        return 'UNOBSERVED', envelope
    return '', envelope


def _unit_xy(vector):
    vector = np.asarray(vector, dtype=float)[:2]
    norm = float(np.linalg.norm(vector))
    return vector/norm if norm > 1e-6 else None


def _cross_xy(axis, delta):
    delta = np.asarray(delta, dtype=float)
    return axis[0]*delta[..., 1]-axis[1]*delta[..., 0]


def _guidance_height(points, guide):
    """Guidance height at the nearest XY projection of each point."""
    if len(guide) < 2:
        return np.full(len(points), float(guide[0, 2]))
    a, b = guide[:-1], guide[1:]
    delta = b[:, :2]-a[:, :2]
    length = np.maximum(np.einsum('ij,ij->i', delta, delta), 1e-12)
    fraction = np.clip(np.einsum('nij,ij->ni', points[:, None, :2]-a[None, :, :2], delta)/length, 0., 1.)
    foot = a[None, :, :2]+fraction[..., None]*delta[None]
    segment = np.argmin(np.linalg.norm(foot-points[:, None, :2], axis=2), axis=1)
    rows = np.arange(len(points))
    return a[segment, 2]+fraction[rows, segment]*(b[segment, 2]-a[segment, 2])


class LocalPlanner:
    """Body-window A* over observed free cells.

    ``side_commitment`` keeps the lateral side chosen around one obstacle in a
    world-fixed frame (origin, axis toward the target at commit time). Fresh
    searches only get a soft cost on the opposite side, so the committed side is
    never forced through blocked or unknown cells; it merely breaks the near-tie
    that otherwise flips left/right on every new observation. When a commitment
    starts and a distance field is available, both sides are searched and the
    one with the larger observed clearance is preferred. ``grid_heading`` =
    ``'target'`` orients the search window toward the current target (with
    hysteresis) instead of the measured yaw, and snaps its cell centres to a
    world lattice of that orientation. With the body-anchored default a few
    centimetres of motion shift every cell, and a marginal gap whose cells sit
    near the clearance limit can open and close between consecutive scans.
    """
    def __init__(self, flight, hz=10., resolution=.4, connect_start=False, route_persistence=False, *,
                 side_commitment=False, side_bias=1.5, side_trigger=.3, side_deadband=.2,
                 side_release=1., side_timeout=20., side_length_ratio=1.6, grid_heading='yaw',
                 heading_hysteresis=.15, height_weight=.5, max_climb_slope=None, grid_forward=3.2,
                 route_refresh_remaining=None, agl_band=None):
        self.flight, self.hz, self.resolution = flight, hz, resolution
        self.connect_start = connect_start
        self.route_persistence = bool(route_persistence)
        # A kept route that ended at its lookahead target is replaced once this
        # little of it remains: each spline stops at the route end, so a short
        # remainder makes it brake (run 31e3924c: a slowdown every 4 m).
        if route_refresh_remaining is not None and not (np.isfinite(route_refresh_remaining)
                                                        and route_refresh_remaining > .5):
            raise ValueError('route_refresh_remaining must exceed the 0.5 m near-end distance')
        self.route_refresh_remaining = None if route_refresh_remaining is None else float(route_refresh_remaining)
        self._route_reached_target = False
        if grid_heading not in ('yaw', 'target'):
            raise ValueError("grid_heading must be 'yaw' or 'target'")
        values = (side_bias, side_trigger, side_deadband, side_release, side_timeout, side_length_ratio)
        if not all(np.isfinite(v) and v >= 0 for v in values) or side_length_ratio < 1.:
            raise ValueError('Side-commitment parameters must be finite and nonnegative')
        self.side_commitment, self.grid_heading = bool(side_commitment), grid_heading
        self.side_bias, self.side_trigger, self.side_deadband = float(side_bias), float(side_trigger), float(side_deadband)
        self.side_release, self.side_timeout = float(side_release), float(side_timeout)
        self.side_length_ratio = float(side_length_ratio)
        if not np.isfinite(heading_hysteresis) or heading_hysteresis < 0:
            raise ValueError('heading_hysteresis must be finite and nonnegative')
        self.heading_hysteresis = float(heading_hysteresis)
        if not np.isfinite(height_weight) or height_weight < 0:
            raise ValueError('height_weight must be finite and nonnegative')
        self.height_weight = float(height_weight)
        if max_climb_slope is not None and (not np.isfinite(max_climb_slope) or max_climb_slope <= 0):
            raise ValueError('max_climb_slope must be positive when enabled')
        self.max_climb_slope = None if max_climb_slope is None else float(max_climb_slope)
        if agl_band is not None and not (np.isfinite(agl_band).all() and 0 < agl_band[0] < agl_band[1]):
            raise ValueError('agl_band must be two ordered positive heights above the ground')
        self.agl_band = None if agl_band is None else tuple(float(v) for v in agl_band)
        self._side = self._heading = None
        self._route = self._route_key = None
        # Each spline stops at the end of its route, so the route length bounds the
        # speed: a 3.2 m window gave ~1.4 m routes and 0.35 m/s at any speed limit
        # (run 99ccbd18). The snapshot (5 m + 1 m) must still cover route+clearance.
        if not np.isfinite(grid_forward) or not 1.6 <= grid_forward <= 4.8:
            raise ValueError('grid_forward must be within 1.6..4.8 m')
        self.lower = np.array([-.8, -2.4, -.4])
        shape = np.rint((np.array([float(grid_forward), 2.4, .8])-self.lower)/resolution).astype(int)+1
        self.cells = np.indices(shape).reshape(3, -1).T
        self.centers = self.lower+self.cells*resolution
        self.shape = tuple(shape)
        self.start = np.rint(-self.lower/resolution).astype(np.int64)
        near = np.meshgrid(np.arange(-.3, .31, .1), np.arange(-.3, .31, .1), [-.1, 0., .1])
        self.near = np.stack(near, axis=-1).reshape(-1, 3)

    def _retained_route(self, position, safety, key, timestamp, target=None):
        """Trim a previous world route, then recheck it against this observation."""
        if self._route is None:
            return None, 'NO_ROUTE'
        if key != self._route_key:
            self._route = None
            return None, 'GOAL_CHANGED'
        route = self._route
        delta = np.diff(route, axis=0)
        length = np.linalg.norm(delta, axis=1)
        arc = np.r_[0., np.cumsum(length)]
        fraction = np.clip(np.sum((position-route[:-1])*delta, axis=1)/length**2, 0., 1.)
        projections = route[:-1]+fraction[:, None]*delta
        index = int(np.argmin(np.linalg.norm(projections-position, axis=1)))
        progress = float(arc[index]+fraction[index]*length[index])
        remaining = arc[-1]-progress
        reason = None
        if np.linalg.norm(projections[index]-position) > .5:
            reason = 'DEVIATION'
        elif timestamp < self._route_progress_time:
            reason = 'CLOCK_REVERSED'
        elif remaining <= (.5 if arc[-1] > .5 else .08):
            reason = 'NEAR_END'
        elif (self.route_refresh_remaining is not None and target is not None and self._route_reached_target
              and remaining <= self.route_refresh_remaining
              and np.linalg.norm(route[-1, :2]-np.asarray(target)[:2]) > ROUTE_TARGET_GAP):
            # Only a lookahead-limited route: one cut short by obstacles or
            # unknown space is kept, so a marginal gap cannot flip it each scan.
            reason = 'EXTEND'
        else:
            if progress-self._route_progress >= .05:
                self._route_progress, self._route_progress_time = progress, timestamp
            if timestamp-self._route_progress_time >= 2.:
                reason = 'NO_PROGRESS'
        if reason is None:
            trimmed = np.vstack([position, projections[index], route[index+1:]])
            trimmed = trimmed[np.r_[True, np.linalg.norm(np.diff(trimmed, axis=0), axis=1) > 1e-7]]
            dense = np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/.04))+1))
                                    for a, b in zip(trimmed[:-1], trimmed[1:])])
            if not safety.clear(dense, start=position):
                reason = 'OBSTACLE_OR_BOUNDS'
            elif not safety.known(dense, position, body_padding=BODY_NUMERIC_TOLERANCE):
                reason = 'UNOBSERVED'
            else:
                return trimmed, ''
        self._route = None
        return None, reason

    def _side_state(self, position, target, timestamp):
        """Return a still-valid side commitment, or end it with an explicit reason."""
        commit = self._side
        if commit is None:
            return None, 'NONE'
        axis = _unit_xy(np.asarray(target)-position)
        reason = None
        if timestamp-commit['time'] > self.side_timeout:
            reason = 'TIMEOUT'
        elif axis is not None and float(axis@commit['axis']) < .5:
            reason = 'DIRECTION_CHANGED'
        elif float((position[:2]-commit['anchor'])@commit['axis']) > self.side_release:
            reason = 'PASSED'
        elif abs(float(_cross_xy(commit['axis'], position[:2]-commit['origin']))) > 3.:
            reason = 'DEVIATION'
        if reason:
            self._side = None
            return None, reason
        return commit, 'COMMITTED'

    def _route_side(self, route, origin, axis):
        """Signed side (+1 left of axis) of the largest lateral excursion, if any."""
        lateral = _cross_xy(axis, np.asarray(route)[:, :2]-np.asarray(origin)[:2])
        index = int(np.argmax(np.abs(lateral)))
        if abs(lateral[index]) < self.side_trigger:
            return 0, None
        return int(np.sign(lateral[index])), np.asarray(route)[index, :2].copy()

    @staticmethod
    def _excursion(route, origin, axis):
        return float(np.abs(_cross_xy(axis, np.asarray(route)[:, :2]-np.asarray(origin)[:2])).max())

    def _advance_anchor(self, commit, route, position):
        """Move the anchor to the farthest strong excursion still AHEAD of the vehicle.

        The vehicle itself sits off the line during a bypass; using the whole
        route would drag the anchor along with it and never release.
        """
        route = np.asarray(route)[:, :2]-commit['origin']
        lateral, along = _cross_xy(commit['axis'], route), route@commit['axis']
        here = float((position[:2]-commit['origin'])@commit['axis'])
        strong = np.abs(lateral) >= max(self.side_trigger, .5*commit['excursion'])
        mask = (along > here+.3) & strong & (np.sign(lateral) == commit['sign'])
        if mask.any():
            index = np.flatnonzero(mask)[np.argmax(along[mask])]
            if along[index] > float((commit['anchor']-commit['origin'])@commit['axis']):
                commit['anchor'] = route[index]+commit['origin']

    def _follow_guidance_height(self, route, guidance_points, safety, position, band=None):
        """Replace 0.4 m cell levels by the (slope-limited) guidance height.

        A* heights come in whole cells, so a 0.3 m terrain rise became a 0.4 m
        step and a steep spline climb into the unobserved space above a
        roll-180 LiDAR (run 401c7d8c). Points within one cell of the guidance
        take its height, then the profile is slope-limited from the vehicle;
        larger deliberate vertical detours stay. The result must pass the same
        clearance, observed-space and ground-band checks, otherwise the A*
        route is kept.
        """
        route, guide = np.asarray(route, float), np.asarray(guidance_points, float).reshape(-1, 3)
        if len(route) < 2 or len(guide) < 2:
            return route
        height = _guidance_height(route, guide)
        smooth = route.copy()
        follow = np.abs(route[:, 2]-height) < self.resolution
        follow[0] = False
        smooth[follow, 2] = height[follow]
        step = np.linalg.norm(np.diff(smooth[:, :2], axis=0), axis=1)*self.max_climb_slope
        for i in range(1, len(smooth)):
            if follow[i]:
                smooth[i, 2] = np.clip(smooth[i, 2], smooth[i-1, 2]-step[i-1], smooth[i-1, 2]+step[i-1])
        dense = np.concatenate([np.linspace(p, q, max(2, int(np.ceil(np.linalg.norm(q-p)/.04))+1))
                                for p, q in zip(smooth[:-1], smooth[1:])])
        if (safety.clear(dense, start=position) and safety.known(dense, position, body_padding=BODY_NUMERIC_TOLERANCE)
                and (band is None or band(smooth[1:]).all())):
            return smooth
        return route

    def _band_limits(self, points, guidance_points, ground):
        """World-Z interval of ``agl_band`` above the ground at each point's own XY.

        Climbs over obstacles reached 3.55 m above the ground against the
        task's 3.5 m (runs bac685bb, 1102eec3): height bounds were absolute.
        The guidance height stays allowed where slope limits keep it outside
        the band on steep terrain; unknown ground sets no bound.
        """
        points = np.asarray(points, float).reshape(-1, 3)
        z, valid = ground(points[:, :2])
        guide = _guidance_height(points, np.asarray(guidance_points, float).reshape(-1, 3))
        low = np.where(valid, np.minimum(z+self.agl_band[0], guide), -np.inf)
        high = np.where(valid, np.maximum(z+self.agl_band[1], guide), np.inf)
        return low, high

    def _within_band(self, points, guidance_points, ground):
        low, high = self._band_limits(points, guidance_points, ground)
        z = np.asarray(points, float).reshape(-1, 3)[:, 2]
        return (z >= low-1e-9) & (z <= high+1e-9)

    @staticmethod
    def _length(route):
        return float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())

    @staticmethod
    def _min_clearance(distance_field, route):
        dense = np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/.1))+1))
                                for a, b in zip(route[:-1], route[1:])]) if len(route) > 1 else np.asarray(route)
        query = distance_field.query(dense)
        values = np.asarray(query.distance)[np.asarray(query.in_bounds)]
        return float(values.min()) if len(values) else -np.inf

    def plan(self, points, evidence, position, yaw, velocity, acceleration, tracker, *, geometry_only=False,
             distance_field=None, clearance_weight=0., preferred_clearance=1.5, planning_reserve=0.,
             timestamp=None, gate_hold=False, ground=None):
        """``ground(xy) -> (z, valid)`` applies ``agl_band`` to every new route."""
        if (not np.isfinite(clearance_weight) or clearance_weight < 0
                or not np.isfinite(preferred_clearance) or preferred_clearance <= 0):
            raise ValueError('Distance-field clearance cost must be finite and nonnegative')
        if self.route_persistence and (timestamp is None or not np.isfinite(timestamp)):
            raise ValueError('Route persistence requires a finite observation timestamp')
        if self.side_commitment and (timestamp is None or not np.isfinite(timestamp)):
            raise ValueError('Side commitment requires a finite observation timestamp')
        position = np.asarray(position, dtype=float)
        heading, offset = yaw, np.zeros(3)
        if self.grid_heading == 'target':
            toward = np.asarray(tracker.target, dtype=float)[:2]-position[:2]
            if np.linalg.norm(toward) > .3:
                bearing = float(np.arctan2(toward[1], toward[0]))
                if (self._heading is None
                        or abs(np.angle(np.exp(1j*(bearing-self._heading)))) > self.heading_hysteresis):
                    self._heading = bearing
            heading = self._heading if self._heading is not None else yaw
            # World-lattice cell centres of this orientation; the start cell is
            # the one containing the vehicle, within half a cell of it. Only XY is
            # snapped: a world-fixed height lattice would pull every route up or
            # down by up to half a cell from the held altitude.
            local = world_to_local(position[None], np.zeros(3), heading)[0]
            offset = np.round(local/self.resolution)*self.resolution-local
            offset[2] = 0.
        centers = self.centers+offset
        side_reason, side_sign, side_alternative = 'DISABLED', 0, None
        guidance = tracker.guidance
        safety = PathSafety(points, evidence, guidance.corridor, self.flight, self.hz, guidance.height_bounds)
        safety.frame_position, safety.frame_yaw = position, yaw
        physical_safety = safety
        safety = safety.with_clearance_reserve(planning_reserve)
        band = (None if self.agl_band is None or ground is None
                else lambda points: self._within_band(points, guidance.points, ground))
        band_removed = 0
        key = (guidance.route_id, guidance.revision, guidance.segment_id, bool(gate_hold))
        retained, persistence_reason = (self._retained_route(position, safety, key, timestamp, tracker.target)
            if self.route_persistence else (None, 'DISABLED'))
        commit = None
        if self.side_commitment:
            commit, side_reason = self._side_state(position, tracker.target, timestamp)
            side_sign = commit['sign'] if commit is not None else 0
        if retained is None:
            world = local_to_world(centers, position, heading)
            # Half a cell keeps every A* edge clear and, beyond that, leaves a
            # buffer above the mission's admission envelope (body + reserve).
            # Without it routes hug that envelope and ordinary tracking error parks
            # the vehicle inside it, where no candidate move is admitted (run
            # 57dd544f: stopped 0.79 m from a trunk against a 0.80 m envelope).
            free = safety.clear_points(world, padding=self.resolution/2)
            free[free] &= safety.known_points(world[free], position, body_padding=BODY_NUMERIC_TOLERANCE)
            if band is not None:
                # The vehicle's own cell stays: above the band it routes back down.
                low, high = self._band_limits(world, guidance.points, ground)
                inside = (world[:, 2] >= low-1e-9) & (world[:, 2] <= high+1e-9)
                inside[np.ravel_multi_index(self.start, self.shape)] = True
                # A cell up to one level outside stands for its point on the band
                # edge when that point is clear and observed; the route is pressed
                # onto the edge below. Over a branch the 0.4 m levels left 3.38 m
                # (too high) and 2.98 m (too low) against a 3.35 m ceiling (run
                # 853c339a: stuck for 100 s).
                edge = ~inside & free & (world[:, 2] <= high+self.resolution) & (world[:, 2] >= low-self.resolution)
                if edge.any():
                    pressed = world[edge].copy()
                    pressed[:, 2] = np.clip(pressed[:, 2], low[edge], high[edge])
                    clear = safety.clear_points(pressed, padding=self.resolution/2)
                    clear[clear] &= safety.known_points(pressed[clear], position, body_padding=BODY_NUMERIC_TOLERANCE)
                    inside[np.flatnonzero(edge)[clear]] = True
                band_removed = int((free & ~inside).sum())
                free &= inside
            free = free.reshape(self.shape)
            # Distance/gradient queries follow the supplied planner's map interface.
            # Distance only ranks already body-clear, observed cells; it never turns
            # unknown space into free space or bypasses independent stopping checks.
            penalty = np.zeros(self.shape, np.float32)
            if distance_field is not None and clearance_weight:
                query = distance_field.query(world)
                deficit = np.clip(1.-query.distance/preferred_clearance, 0., 1.)
                deficit[~query.in_bounds] = 1.
                penalty = (clearance_weight*deficit**2).reshape(self.shape).astype(np.float32)
            if self.height_weight:
                # Climbing early or late costs the same, so A* used to climb on its
                # last step; near the goal the gap then closes almost vertically,
                # into the cone above the sensor that a 52 deg LiDAR never scans
                # (run 3a2722fb: REFERENCE_UNOBSERVED, uncertified stop). Pay for
                # height error per step so the climb happens while moving forward.
                error = np.minimum(np.abs(world[:, 2]-float(tracker.target[2])), 1.)/self.resolution
                penalty = penalty+(self.height_weight*error).reshape(self.shape).astype(np.float32)
            desired, route, blocked = np.zeros(3), np.array([position]), 'UNOBSERVED'
            start = self.start if free[tuple(self.start)] else None
            connected, _ = label(free)
            if start is None and self.connect_start:
                # The actual body may be clear while a whole coarse cell around it is not.
                # Connect to a real free cell through an explicitly checked observed segment;
                # never turn the blocked start cell into free space.
                distance = np.linalg.norm(world-position, axis=1)
                candidates = np.flatnonzero(free.ravel() & (distance <= 1.2))
                candidates = candidates[np.argsort(distance[candidates])[:12]]
                best = np.inf
                checked_components = set()
                for index, clear in zip(candidates, safety.clear_segments(position, world[candidates], leaving=True)):
                    component = connected.ravel()[index]
                    if component in checked_components or not clear:
                        continue
                    segment = np.linspace(position, world[index], max(2, int(np.ceil(distance[index]/.04))+1))
                    if not safety.known(segment, position, body_padding=BODY_NUMERIC_TOLERANCE):
                        continue
                    checked_components.add(component)
                    reachable = connected.ravel() == component
                    score = np.linalg.norm(world[reachable]-tracker.target, axis=1).min()+.05*distance[index]
                    if score < best:
                        best, start = score, self.cells[index].astype(np.int64)
            if start is not None:
                reachable = connected.ravel() == connected[tuple(start)]
                base_score = np.linalg.norm(world-tracker.target, axis=1)
                base_score[~reachable] = np.inf

                def search(opposite=None):
                    # A soft cost only: blocked/unknown cells stay excluded by ``free``.
                    cost, score = penalty, base_score
                    if opposite is not None:
                        cost = penalty+(self.side_bias*opposite).reshape(self.shape).astype(np.float32)
                        score = base_score+.5*opposite
                    end = self.cells[np.argmin(score)].astype(np.int64)
                    path = astar(free, cost, start, end)
                    found = local_to_world(self.lower+offset+np.asarray(path)*self.resolution, position, heading)
                    if band is not None:
                        low, high = self._band_limits(found, guidance.points, ground)
                        found[:, 2] = np.clip(found[:, 2], low, high)
                    if not np.array_equal(start, self.start):
                        return np.vstack([position, found])
                    # The start cell contains the vehicle; its lattice centre can lie up
                    # to half a cell behind it. Starting from that centre puts a backward
                    # kink in the route (run 4e5b0c28), so start from the vehicle itself.
                    found[0] = position
                    return found

                if self.side_commitment:
                    if commit is not None:
                        lateral = _cross_xy(commit['axis'], world[:, :2]-commit['origin'])
                        route = search(commit['sign']*lateral < -self.side_deadband)
                        sign, anchor = self._route_side(route, commit['origin'], commit['axis'])
                        if sign == commit['sign']:
                            commit['time'] = float(timestamp)
                            self._advance_anchor(commit, route, position)
                        elif sign == -commit['sign']:
                            # The soft cost was not enough: the committed side is not
                            # reachable in this observation. Commit to the side that is,
                            # so a later reopening of the old side does not flip back.
                            commit.update(sign=sign, anchor=anchor, time=float(timestamp),
                                          excursion=self._excursion(route, commit['origin'], commit['axis']))
                            side_reason = 'SWITCHED_BLOCKED'
                        side_sign = commit['sign']
                    else:
                        route = search()
                        axis = _unit_xy(np.asarray(tracker.target)-position)
                        sign, anchor = self._route_side(route, position, axis) if axis is not None else (0, None)
                        if sign:
                            if distance_field is not None:
                                # Search the other side as well; prefer the larger observed clearance.
                                lateral = _cross_xy(axis, world[:, :2]-position[:2])
                                try:
                                    alternative = search(sign*lateral > self.side_deadband)
                                except ValueError:
                                    alternative = None
                                if alternative is not None:
                                    other, other_anchor = self._route_side(alternative, position, axis)
                                    if (other == -sign and self._length(alternative)
                                            <= self.side_length_ratio*max(self._length(route), 1e-6)):
                                        mine = self._min_clearance(distance_field, route)
                                        theirs = self._min_clearance(distance_field, alternative)
                                        side_alternative = dict(sign=other, clearance_m=theirs, chosen_clearance_m=mine)
                                        if theirs > mine+.05:
                                            route, sign, anchor = alternative, other, other_anchor
                                            side_alternative['chosen_clearance_m'] = theirs
                            self._side = dict(sign=sign, origin=position[:2].copy(), axis=axis,
                                              anchor=anchor, time=float(timestamp),
                                              excursion=self._excursion(route, position, axis))
                            side_reason, side_sign = 'NEW', sign
                else:
                    route = search()
                if self.max_climb_slope is not None:
                    route = self._follow_guidance_height(route, guidance.points, safety, position, band)
                for target in route[1:4][::-1]:
                    if safety.clear([position, target], start=position):
                        delta = target-position
                        desired = delta/max(np.linalg.norm(delta), .01)*min(guidance.speed_limit, self.flight.max_speed,
                                                                           1.5*np.linalg.norm(delta))
                        blocked = ''
                        break
            if not np.linalg.norm(desired):
                # 粗网格无法表达近处小步移动时，用已观测的短步继续获取观测。
                nearby = local_to_world(self.near, position, heading)
                geometric = safety.clear_points(nearby)
                free_near = geometric.copy()
                free_near[free_near] &= safety.known_points(nearby[free_near], position, body_padding=BODY_NUMERIC_TOLERANCE)
                if band is not None:
                    free_near &= band(nearby)
                candidates = nearby[free_near]
                score = np.linalg.norm(candidates-tracker.target, axis=1)
                if commit is not None:
                    # Same soft side preference as the grid search; safety checks unchanged.
                    side = commit['sign']*_cross_xy(commit['axis'], candidates[:, :2]-commit['origin'])
                    score = score+.5*(side < -self.side_deadband)
                # The nearest three can share XY and fail the same segment check.
                # Search the bounded 7x7x3 set so a safe lateral step stays usable.
                ordered = candidates[np.argsort(score)]
                length = np.linalg.norm(ordered-position, axis=1)
                ordered = ordered[(length >= .05) & (np.linalg.norm(ordered-tracker.target, axis=1)
                                                     < np.linalg.norm(position-tracker.target))]
                clear = safety.clear_segments(position, ordered, leaving=True)
                if clear.any():
                    target = ordered[np.argmax(clear)]
                    delta = target-position
                    length = np.linalg.norm(delta)
                    desired = delta/length*min(.4, guidance.speed_limit, self.flight.max_speed, 1.5*length)
                    route, blocked = np.array([position, target]), ''
                toward = np.linalg.norm(nearby-tracker.target, axis=1) < np.linalg.norm(position-tracker.target)-.05
                if not np.linalg.norm(desired) and not (toward & geometric).any():
                    blocked = 'OBSTACLE'
        else:
            route, blocked = retained, ''
            if commit is not None:
                commit['time'] = float(timestamp)
            desired = np.zeros(3)
            penalty = np.zeros(self.shape, np.float32)
            for target in route[1:4][::-1]:
                if safety.clear([position, target], start=position):
                    delta = target-position
                    distance = np.linalg.norm(delta)
                    desired = delta/max(distance, .01)*min(self._route_speed, 1.5*distance)
                    break
        if self.route_persistence and retained is None and len(route) > 1 and np.linalg.norm(desired) > 1e-8:
            self._route = route[np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-7]].copy()
            self._route_key, self._route_progress_time = key, float(timestamp)
            self._route_progress, self._route_speed = 0., float(np.linalg.norm(desired))
            self._route_reached_target = bool(
                np.linalg.norm(self._route[-1, :2]-np.asarray(tracker.target)[:2]) <= ROUTE_TARGET_GAP)
        raw = desired.copy()
        if geometry_only:
            # A spline executor owns dynamics and braking. Do not construct an
            # unused velocity command from a possibly out-of-limit measured state.
            metrics = dict(esdf_used=distance_field is not None,
                           esdf_astar_penalty_max=float(penalty.max()),
                           planning_clearance_reserve_m=float(planning_reserve),
                           route_reused=retained is not None, persistence_reset_reason=persistence_reason,
                           side_commitment=side_reason, side_sign=int(side_sign),
                           side_alternative=side_alternative, grid_heading_rad=float(heading),
                           agl_band_removed_cells=band_removed)
            if distance_field is not None:
                metrics['esdf_route_min_distance_m'] = float(distance_field.query(route).distance.min())
            return route, dict(raw_velocity=raw.tolist(), reason=blocked, route=route.tolist(), **metrics), physical_safety
        first_reason = ''
        for scale in (1., .5, .25, 0.):
            command = VelocityTrajectory(position, velocity, acceleration, scale*desired, self.flight)
            reason, envelope = certificate(safety, command, position, 2/self.hz)
            if not reason:
                if scale == 0. and (first_reason == 'UNOBSERVED' or blocked == 'UNOBSERVED'):
                    direction = 1. if tracker.target[2] >= position[2] else -1.
                    for sign in (direction, -direction):
                        probe = np.array([0., 0., sign*min(.15, guidance.speed_limit)])
                        observation = VelocityTrajectory(position, velocity, acceleration, probe, self.flight)
                        rejection, observed_envelope = certificate(safety, observation, position, 2/self.hz)
                        if not rejection:
                            return observation, dict(raw_velocity=raw.tolist(), executed_velocity=probe.tolist(),
                                reason='', certified=True, shield_intervened=True, shield_reason='UNOBSERVED',
                                action='OBSERVE', route=route.tolist(), envelope=observed_envelope.tolist()), physical_safety
                return command, dict(raw_velocity=raw.tolist(), executed_velocity=(scale*desired).tolist(),
                                     reason=blocked or (first_reason if scale == 0. else ''),
                                     certified=True, shield_intervened=scale != 1., shield_reason=first_reason,
                                     action='MOVE' if np.linalg.norm(scale*desired) > 1e-8 else 'BRAKE',
                                     route=route.tolist(), envelope=envelope.tolist()), physical_safety
            first_reason = first_reason or reason
        brake = QuinticPath.brake(position, velocity, acceleration, self.flight.max_acceleration, self.flight.max_jerk)
        envelope = brake.remaining(0., position)
        certified = safety.clear(envelope) and safety.known(envelope, position, body_padding=BODY_NUMERIC_TOLERANCE)
        return brake, dict(raw_velocity=raw.tolist(), executed_velocity=[0., 0., 0.], reason=reason,
                           certified=certified, shield_intervened=bool(np.linalg.norm(raw)),
                                 action='BRAKE', route=route.tolist(), envelope=envelope.tolist()), physical_safety
