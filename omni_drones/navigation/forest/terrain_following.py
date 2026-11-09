"""Observation-only height surface and checked XY-to-XYZ planning.

This is an offline/runtime building block, not a replacement for the flight
controller's live observation guard. The lower-surface estimator is deliberately
not called CSF: it uses supported local minima and bounded triangulation.
"""
from dataclasses import dataclass
import json
import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree
from .geometry import BOUNDS, Z_BOUNDS, collision_free, observed_centers

PROTOCOL = 'dual_view_surface_xy_v1'


def supported_triangles(vertices, triangulation, max_edge, max_slope):
    """Reject degenerate simplices individually without losing the surface."""
    points = np.asarray(vertices, float)[triangulation.simplices]
    delta = points[:, 1:]-points[:, :1]
    xy = delta[:, :, :2]
    determinant = xy[:, 0, 0]*xy[:, 1, 1]-xy[:, 0, 1]*xy[:, 1, 0]
    scale = np.sum(xy*xy, axis=(1, 2))
    valid = (np.abs(determinant) > 1e-8*np.maximum(scale, 1e-12))
    valid &= np.isfinite(points).all(axis=(1, 2))
    valid &= np.isfinite(triangulation.transform).all(axis=(1, 2))
    slopes = np.full((len(points), 2), np.inf)
    ids = np.flatnonzero(valid)
    slopes[ids, 0] = (delta[ids, 0, 2]*xy[ids, 1, 1]-xy[ids, 0, 1]*delta[ids, 1, 2])/determinant[ids]
    slopes[ids, 1] = (xy[ids, 0, 0]*delta[ids, 1, 2]-delta[ids, 0, 2]*xy[ids, 1, 0])/determinant[ids]
    edges = points[:, [1, 2, 0], :2]-points[:, :, :2]
    return (valid & (np.linalg.norm(edges, axis=2).max(axis=1) <= max_edge)
            & (np.linalg.norm(slopes, axis=1) <= max_slope))


@dataclass
class GroundQuery:
    z: np.ndarray
    valid: np.ndarray
    age_s: np.ndarray
    confidence: np.ndarray  # Geometric support score, NOT a calibrated probability.


class GroundSurface:
    """No extrapolation over missing ground, cliffs, old or isolated returns.

    Inputs are gravity-aligned local *raw* returns (not the cropped obstacle
    volume). A dense canopy can still resemble a lower surface; this estimator
    does not certify semantic ground. Flight always needs independent 3D checks.
    """
    def __init__(self, points, times, now, cell=.5, max_edge=1.6,
                 max_slope=.6, max_age=60.):
        self.tri = None
        self.max_age = max_age
        p = np.asarray(points, float).reshape(-1, 3)
        t = np.broadcast_to(np.asarray(times, float), (len(p),))
        good = np.isfinite(p).all(axis=1) & np.isfinite(t) & (t <= now) & (t >= now-max_age)
        p, t = p[good], t[good]
        self.vertices = np.empty((0, 3))
        if len(p) < 12:
            return
        _, inv = np.unique(np.floor(p[:, :2]/cell).astype(int), axis=0, return_inverse=True)
        order = np.argsort(inv)
        groups = np.split(order, np.flatnonzero(np.diff(inv[order]))+1)
        vertices, ages = [], []
        for ids in groups:
            if len(ids) < 3:
                continue
            low = np.quantile(p[ids, 2], .1)
            ids = ids[np.abs(p[ids, 2]-low) <= .12]
            if len(ids) < 3:
                continue
            vertices.append(np.median(p[ids], axis=0))
            ages.append(float(now-np.median(t[ids])))
        v = np.asarray(vertices).reshape(-1, 3)
        if len(v) < 6:
            return
        # Suppress elevated isolated columns using spatially distributed support.
        dist, ids = cKDTree(v[:, :2]).query(v[:, :2], k=min(12, len(v)))
        values = np.where(dist <= 1.6, v[ids, 2], np.nan)
        keep = (np.sum(dist <= 1.6, axis=1) >= 6) & (v[:, 2] <= np.nanpercentile(values, 25, axis=1)+.4)
        self.vertices = v[keep]
        self.ages = np.asarray(ages)[keep]
        if len(self.vertices) < 6:
            return
        try:
            tri = Delaunay(self.vertices[:, :2])
        except QhullError:
            return
        self.triangle_valid = supported_triangles(self.vertices, tri, max_edge, max_slope)
        self.tri = tri

    def query(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        z, age = np.full(len(xy), np.nan), np.full(len(xy), np.inf)
        valid, confidence = np.zeros(len(xy), bool), np.zeros(len(xy))
        if self.tri is not None:
            sid = self.tri.find_simplex(xy)
            ids = np.flatnonzero((sid >= 0) & np.isfinite(xy).all(axis=1))
            ids = ids[self.triangle_valid[sid[ids]]]
            s = sid[ids]
            uv = np.einsum('nij,nj->ni', self.tri.transform[s, :2], xy[ids]-self.tri.transform[s, 2])
            weights = np.c_[uv, 1-uv.sum(axis=1)]
            vertices = self.tri.simplices[s]
            z[ids] = (weights*self.vertices[vertices, 2]).sum(axis=1)
            age[ids] = self.ages[vertices].max(axis=1)
            valid[ids] = True
            confidence[ids] = np.clip(1-age[ids]/self.max_age, 0, 1)
        return GroundQuery(z, valid, age, confidence)


class HeightReference:
    """One height surface per request; no per-cell switch to a flat plane."""
    def __init__(self, ground, desired_agl=2.5, anchor_tolerance=.35):
        if not np.isfinite(desired_agl) or desired_agl <= 0:
            raise ValueError('positive desired AGL required')
        self.ground, self.desired_agl = ground, float(desired_agl)
        anchor = ground.query([[0., 0.]])
        self.offset = float(anchor.z[0]+desired_agl) if anchor.valid[0] else float('nan')
        self.mode = ('terrain_following' if anchor.valid[0] and abs(self.offset) <= anchor_tolerance
                     else 'hold_current_altitude_ground_unreliable')

    def query(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.mode != 'terrain_following':
            return np.zeros(len(xy)), np.isfinite(xy).all(axis=1)
        q = self.ground.query(xy)
        # Smoothly remove the small initial AGL offset; never jump from z=0.
        blend = np.exp(-np.sum(xy*xy, axis=1)/2.)
        z = q.z+self.desired_agl-self.offset*blend
        return z, q.valid


def densify(points, spacing=.04):
    points = np.asarray(points, float)
    if len(points) < 2:
        return points.copy()
    return np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/spacing))+1))
                           for a, b in zip(points[:-1], points[1:])])


def surface_grid(reference, points, evidence, resolution=.2):
    """Sampled BEV, like the original data: classify each cell's center.

    A pixel is not a continuous-volume safety certificate. audit_path separately
    checks every path segment and the smoothed trajectory in 3D, as well as
    requiring their displayed pixels to be free.
    """
    xs = np.arange(BOUNDS[0]+resolution/2, BOUNDS[1], resolution)
    ys = np.arange(BOUNDS[2]+resolution/2, BOUNDS[3], resolution)
    xy = np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1).reshape(-1, 2)
    obstacle_tree, evidence_tree = cKDTree(points), cKDTree(evidence)
    z, valid_all = reference.query(xy)
    centers = np.c_[xy, np.nan_to_num(z)]
    valid_all &= (centers[:, 2] >= Z_BOUNDS[0]) & (centers[:, 2] <= Z_BOUNDS[1])
    clear_all = collision_free(obstacle_tree, centers, padding=.02, clearance=.75)
    known_all = observed_centers(evidence_tree, centers, actual=np.zeros(3))
    state = np.zeros(len(xy), np.uint8)
    state[valid_all & clear_all & known_all] = 1
    state[valid_all & ~clear_all] = 2
    return xs, ys, state.reshape(len(xs), len(ys))


def raster_free(xy, grid, resolution=.2):
    xs, ys, state = grid
    xy = np.asarray(xy).reshape(-1, 2)
    ij = np.floor((xy-[BOUNDS[0], BOUNDS[2]])/resolution).astype(int)
    valid = (ij >= 0).all(axis=1) & (ij < state.shape).all(axis=1)
    out = np.zeros(len(xy), bool)
    out[valid] = state[ij[valid, 0], ij[valid, 1]] == 1
    return out


def lift_path(xy, reference, spacing=.2):
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('finite XY array required')
    path = densify(np.vstack([[0., 0.], xy]), spacing)
    z, valid = reference.query(path)
    if not valid.all():
        raise ValueError('ground support missing along path')
    return np.c_[path, z]


def audit_path(xy, reference, safety, velocity, grid=None, dynamics=True):
    """Never repair a model path silently. Check its lift, smoothing and brakes."""
    result = dict(accepted=False, height_mode=reference.mode)
    try:
        path = lift_path(xy, reference)
        dense = densify(path)
        result.update(obstacle_clear=safety.clear(dense), coverage_known=safety.known(dense, np.zeros(3)),
                      image_free=bool(grid is None or raster_free(dense[:, :2], grid).all()))
        if not all(result[k] for k in ('obstacle_clear', 'coverage_known', 'image_free')):
            result['reason'] = 'lifted_path_or_displayed_cell_rejected'
            return result, None
        if not dynamics:
            result.update(accepted=True, reason='geometry_only')
            return result, None
        curve = safety.reconstruct(path, np.asarray(velocity), np.zeros(3), np.zeros(3))
        ts = np.unique(np.r_[np.linspace(0., curve.duration, max(101, int(curve.duration/.03)+1)), curve.times])
        p, v, a = curve.sample(ts)
        target, supported = reference.query(p[:, :2])
        result.update(smooth_clear=safety.clear(p), smooth_known=safety.known(p, np.zeros(3)),
                      smooth_image_free=bool(grid is None or raster_free(p[:, :2], grid).all()),
                      surface_supported=bool(supported.all()),
                      height_error_m=float(np.max(np.abs(p[:, 2]-target))) if supported.all() else None,
                      max_vz_mps=float(np.abs(v[:, 2]).max()), max_az_mps2=float(np.abs(a[:, 2]).max()),
                      duration_s=float(curve.duration))
        flags = ('smooth_clear', 'smooth_known', 'smooth_image_free', 'surface_supported')
        if not all(result[k] for k in flags) or result['height_error_m'] > .2 or result['max_vz_mps'] > .6 or result['max_az_mps2'] > .6:
            result['reason'] = 'continuous_height_or_safety_rejected'
            return result, None
        # Offline checks over this frozen observation. Online guard remains required.
        # At a future rollout time the vehicle is at p(t), not the request's
        # origin. Using the origin inserts a fictitious shortcut across turns.
        result['braking_safe'] = all(safety.guard(curve, t, curve.sample(t)[0], body_padding=0.)
                                     for t in np.linspace(0., curve.duration, max(2, int(curve.duration*10)+1)))
        result['accepted'] = result['braking_safe']
        result['reason'] = 'accepted' if result['accepted'] else 'braking_rejected'
        return result, curve if result['accepted'] else None
    except ValueError as exc:
        result['reason'] = str(exc)
        return result, None


def parse_xy(text):
    obj = json.loads(text)
    if obj == {'status': 'hold', 'positions_local_xy_m': []}:
        return np.empty((0, 2))
    if not isinstance(obj, dict) or set(obj) != {'positions_local_xy_m'}:
        raise ValueError('exact XY output schema required')
    values = obj['positions_local_xy_m']
    if not isinstance(values, list) or len(values) != 6 or any(
            not isinstance(p, list) or len(p) != 2 or any(type(v) not in (int, float) for v in p) for p in values):
        raise ValueError('six numeric XY points required')
    xy = np.asarray(values, float)
    if not np.isfinite(xy).all() or (np.linalg.norm(xy, axis=1) < 1e-6).any():
        raise ValueError('finite non-origin waypoints required')
    return xy


def prompt(state, reference):
    height = ('本图表示随已估计地面变化的参考飞行高度上的通行情况。' if reference.mode == 'terrain_following'
              else '地面高度依据不足，本图表示暂时保持当前高度时的通行情况。')
    return ('你是林间无人机局部路径规划器。沿当前通道继续蛇形巡检，遇障碍绕行，条件允许时返回原通道。'
        '只有已确认行末且换行连接区域已观测可通行时才换行；障碍、树间缺口和树行虚线端点不等于行末。\n'
        '图1是全局观测：世界XY坐标，红点为当前位置，箭头为朝向，蓝线为已飞轨迹；树行仅为估计，空白不代表自由空间。\n'
        '图2是单层局部BEV：原点为当前机体中心，x向前、y向左；图中向右为+x，向上为+y，单位米。'
        '范围x∈[-2,10]、y∈[-6,6]。'+height+'高度由独立规划器处理，你只输出XY。\n'
        '橙色区域禁止进入。全部航点以及从原点开始的全部连接线，都不得进入、穿越或触碰橙色区域。'
        '浅色为已观测可通行区域。灰色表示尚未确认，可能因遮挡未被观测，不等同于障碍；可作为后续探索方向，'
        '本次执行段先到浅色区域内的安全观察位置，留出未知边界前的停止余量，取得新观测后再延伸。不能假定灰色或图外区域可飞行。\n'
        '当前观测状态：'+json.dumps(state, ensure_ascii=False)+'\n'
        '自由选择6个按行进顺序排列的局部XY航点，不包含原点；允许绕行和转弯，不要求x递增，不限制总长度。'
        '全部连接线须处于浅色区域，避免急折返，不能为凑六点进入未知空间。'
        '只输出JSON：{"positions_local_xy_m":[[x,y],...]}。'
        '找不到安全路径则输出{"status":"hold","positions_local_xy_m":[]}。')
