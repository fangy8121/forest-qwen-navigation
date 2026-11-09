"""米制、重力对齐的局部几何；不依赖 Isaac。"""
import json
import math
import heapq
import numpy as np
from scipy.spatial import cKDTree

RADIUS = 0.5
MARGIN = 0.15
INFLATION = RADIUS + MARGIN
BOUNDS = (-2.0, 10.0, -6.0, 6.0)  # xmin, xmax, ymin, ymax
Z_BOUNDS = (-1.5, 1.5)
HEIGHT_LEVELS = (-.5, 0.0, .5, 1.0, 1.5)  # 展示中心高度；三维规划范围独立由 Z_BOUNDS 定义
GRID_RESOLUTION = 0.2
BODY_HALF_HEIGHT = 0.055  # 仿真碰撞体实测水平半高
BODY_VERTICAL_RADIUS = 0.25  # 初始保守姿态包络；不是实测尺寸，也未由参考加速度推导
VERTICAL_RADIUS = BODY_VERTICAL_RADIUS + MARGIN
PROTOCOL_VERSION = 3
REPRESENTATION = 'height_slices_xyz_v2'
EVIDENCE_TOLERANCE = np.sqrt(3)*GRID_RESOLUTION/2 + .025
REFERENCE_CLEARANCE = .75  # Center-to-surface distance used by map generation.
REFERENCE_PROFILE = 'reference_sphere_v1'


def center_clearance(flight):
    """Versioned geometry; existing adapters retain their cylinder semantics."""
    profile = (flight.get('safety_profile') if hasattr(flight, 'get')
               else getattr(flight, 'safety_profile', None))
    if profile in (None, 'cylinder_xyz_v1'):
        return None
    if profile == REFERENCE_PROFILE:
        return REFERENCE_CLEARANCE
    raise ValueError('unknown safety profile: '+str(profile))


def collision_free(points, centers, padding=0., clearance=None):
    tree = points if isinstance(points, cKDTree) else cKDTree(np.asarray(points).reshape(-1, 3))
    centers = np.asarray(centers).reshape(-1, 3)
    if clearance is not None:
        return tree.query(centers)[0] > clearance+padding
    return ~occupied_centers(tree, centers, INFLATION+padding, VERTICAL_RADIUS+padding)


def occupied_centers(points, centers, horizontal=INFLATION, vertical=VERTICAL_RADIUS):
    """圆柱机体包络与命中点相交；分块限制邻域查询的内存。"""
    tree = points if isinstance(points, cKDTree) else cKDTree(np.asarray(points).reshape(-1, 3))
    centers = np.asarray(centers).reshape(-1, 3)
    occupied = np.zeros(len(centers), dtype=bool)
    if not tree.n or not len(centers):
        return occupied
    extent = np.array([horizontal, horizontal, vertical])
    nearby = ((tree.data >= centers.min(axis=0)-extent) &
              (tree.data <= centers.max(axis=0)+extent)).all(axis=1)
    if np.count_nonzero(nearby) < tree.n/2:
        tree = cKDTree(tree.data[nearby])
        if not tree.n:
            return occupied
    distance, nearest = tree.query(centers)
    delta = tree.data[nearest]-centers
    occupied = (np.linalg.norm(delta[:, :2], axis=1) <= horizontal) & (np.abs(delta[:, 2]) <= vertical)
    # 最近点已在圆柱内或整个外接球为空时，无需展开全部邻居。
    uncertain = np.flatnonzero(~occupied & (distance <= np.hypot(horizontal, vertical)))
    for start in range(0, len(uncertain), 256):
        indices = uncertain[start:start+256]
        batch = centers[indices]
        neighbors = tree.query_ball_point(batch, np.hypot(horizontal, vertical))
        counts = np.fromiter(map(len, neighbors), dtype=int, count=len(batch))
        if not counts.sum():
            continue
        rows = np.repeat(np.arange(len(batch)), counts)
        delta = tree.data[np.concatenate(neighbors).astype(int)] - batch[rows]
        hit = (np.linalg.norm(delta[:, :2], axis=1) <= horizontal) & (np.abs(delta[:, 2]) <= vertical)
        occupied[indices[np.unique(rows[hit])]] = True
    return occupied


def clearance_margin(points, centers, clearance=None, cap=2.):
    """最大的 padding p，使 collision_free(points, centers, p, clearance) 仍成立；至多 cap。

    圆柱包络下取所有邻点 max(水平距-INFLATION, |竖直差|-VERTICAL_RADIUS) 的最小值，
    与 occupied_centers 的判据一致：margin > p 当且仅当该中心在 padding p 下无碰撞。
    """
    tree = points if isinstance(points, cKDTree) else cKDTree(np.asarray(points).reshape(-1, 3))
    centers = np.asarray(centers, float).reshape(-1, 3)
    margin = np.full(len(centers), float(cap))
    if not tree.n or not len(centers):
        return margin
    if clearance is not None:
        return np.minimum(margin, tree.query(centers)[0]-clearance)
    neighbors = tree.query_ball_point(centers, np.hypot(INFLATION+cap, VERTICAL_RADIUS+cap))
    for i, indices in enumerate(neighbors):
        if indices:
            delta = tree.data[indices]-centers[i]
            margin[i] = min(cap, float(np.maximum(np.linalg.norm(delta[:, :2], axis=1)-INFLATION,
                                                  np.abs(delta[:, 2])-VERTICAL_RADIUS).min()))
    return margin


def observed_body_height(max_acceleration=1.):
    # 参考运动的名义扫掠上界；实际跟踪姿态及定位误差仍需独立审计。
    # ||a|| <= a_max 时，重力补偿后的推力倾角满足 sin(theta) <= a_max/g。
    height = BODY_HALF_HEIGHT + RADIUS*float(max_acceleration)/9.81
    if not BODY_HALF_HEIGHT <= height <= BODY_VERTICAL_RADIUS:
        raise ValueError('reference acceleration exceeds the configured body envelope')
    return height


def body_offsets(vertical=None):
    vertical = observed_body_height() if vertical is None else vertical
    axis = np.linspace(-RADIUS, RADIUS, 6)
    x, y = np.meshgrid(axis, axis)
    xy = np.c_[x.ravel(), y.ravel()]
    xy = xy[np.linalg.norm(xy, axis=1) <= RADIUS]
    angles = np.arange(16)*2*np.pi/16
    xy = np.vstack([[0., 0.], xy, RADIUS*np.c_[np.cos(angles), np.sin(angles)]])
    return np.concatenate([np.c_[xy, np.full(len(xy), z)] for z in (-vertical, 0., vertical)])


def observed_centers(evidence, centers, actual=None, vertical=None, body_padding=.05):
    """机体体积采样点均有三维射线证据，才将中心位置标为已观测。"""
    tree = evidence if isinstance(evidence, cKDTree) else cKDTree(np.asarray(evidence).reshape(-1, 3))
    centers = np.asarray(centers).reshape(-1, 3)
    known = np.zeros(len(centers), dtype=bool)
    vertical = observed_body_height() if vertical is None else vertical
    offsets = body_offsets(vertical)
    if tree.n:
        known[:] = True
        # 下表面尚未观测的中心无需继续查询中层和上表面；语义仍是全部采样点有证据。
        for band in np.split(offsets, 3):
            active = np.flatnonzero(known)
            for start in range(0, len(active), 256):
                indices = active[start:start+256]
                samples = centers[indices, None, :] + band
                covered = tree.query(samples.reshape(-1, 3), distance_upper_bound=EVIDENCE_TOLERANCE+1e-12)[0].reshape(samples.shape[:2]) <= EVIDENCE_TOLERANCE
                if actual is not None:
                    delta = samples-np.asarray(actual)
                    covered |= ((np.linalg.norm(delta[..., :2], axis=-1) <= RADIUS+body_padding+1e-12) &
                                (np.abs(delta[..., 2]) <= vertical+body_padding+1e-12))
                known[indices] = covered.all(axis=1)
    return known


def astar_grid(blocked, start, goal):
    """八邻域 A*；输入仅是调用者提供的占据栅格，不访问场景或观测。"""
    start, goal = tuple(start), tuple(goal)
    if blocked[start] or blocked[goal]:
        return None
    queue, costs, parent = [(0., start)], {start: 0.}, {}
    while queue:
        _, u = heapq.heappop(queue)
        if u == goal:
            route = [u]
            while u != start:
                u = parent[u]
                route.append(u)
            return np.asarray(route[::-1])
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
            v = u[0]+dx, u[1]+dy
            if not (0 <= v[0] < blocked.shape[0] and 0 <= v[1] < blocked.shape[1]) or blocked[v]:
                continue
            # 对角移动不能从两个障碍栅格的夹角穿过去。
            if dx and dy and (blocked[u[0]+dx, u[1]] or blocked[u[0], u[1]+dy]):
                continue
            cost = costs[u]+math.hypot(dx, dy)
            if cost < costs.get(v, float('inf')):
                costs[v], parent[v] = cost, u
                heapq.heappush(queue, (cost+math.hypot(v[0]-goal[0], v[1]-goal[1]), v))
    return None


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def world_to_local(points, position, yaw):
    # 行向量右乘 R；仅消除偏航，x 前、y 左、z 上，不随机体滚转。
    return (np.asarray(points) - position) @ yaw_matrix(yaw)


def local_to_world(points, position, yaw):
    return np.asarray(points) @ yaw_matrix(yaw).T + position


def quat_matrix(q):
    # Isaac 四元数顺序为 wxyz；传感器射线必须使用完整姿态。
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def yaw_from_quat(q):
    r = quat_matrix(q)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def segment_distances(points, a, b):
    p, a, b = np.asarray(points), np.asarray(a), np.asarray(b)
    d = b - a
    t = np.clip(((p-a) @ d) / max(float(d @ d), 1e-20), 0, 1)
    return np.linalg.norm(p - (a + t[:, None]*d), axis=1)


def clearance(path, obstacles, radii=0.0):
    """所有线段到障碍表面的最小距离；path 必须显式包含实际起点。"""
    p, o = np.asarray(path), np.asarray(obstacles)
    if not len(o):
        return float('inf')
    return float(min(np.min(segment_distances(o, a, b) - radii)
                     for a, b in zip(p[:-1], p[1:])))


def inside(points, bounds=BOUNDS):
    p = np.asarray(points)
    return bool(np.all((p[:, 0] >= bounds[0]) & (p[:, 0] <= bounds[1]) &
                       (p[:, 1] >= bounds[2]) & (p[:, 1] <= bounds[3])))


def _unique_object(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError('duplicate JSON key')
        d[k] = v
    return d


def corridor_polygon(bounds, position, yaw):
    x0,x1,y0,y1=bounds
    return world_to_local(np.array([[x0,y0,position[2]],[x1,y0,position[2]],
                                    [x1,y1,position[2]],[x0,y1,position[2]]]),position,yaw)[:,:2]


def inside_polygon(points, polygon):
    edges=np.roll(polygon,-1,axis=0)-polygon
    delta=np.asarray(points)[:,None,:]-polygon
    cross=edges[None,:,0]*delta[:,:,1]-edges[None,:,1]*delta[:,:,0]
    return (cross>=-1e-8).all(axis=1) | (cross<=1e-8).all(axis=1)


def parse_path(text, bounds=BOUNDS, protocol_version=PROTOCOL_VERSION):
    # 不截取代码块、不修复 JSON、不补点；布尔值不是坐标数字。
    if protocol_version not in (1, 2, 3):
        raise ValueError('unsupported path protocol')
    try:
        obj = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError('invalid JSON') from e
    if not isinstance(obj, dict) or set(obj) != {'status', 'points'}:
        raise ValueError('exact fields status and points required')
    if obj['status'] in (('HOLD',) if protocol_version == 3 else ('HOLD', 'STOP')) and obj['points'] == []:
        return obj
    if obj['status'] != 'GO' or not isinstance(obj['points'], list) or len(obj['points']) != 6:
        raise ValueError('GO requires exactly six points; HOLD requires []')
    dimensions = 3 if protocol_version == 3 else 2
    for p in obj['points']:
        if not isinstance(p, list) or len(p) != dimensions:
            raise ValueError(f'finite numeric {dimensions}D points required')
        # 先限制数量级，防止 JSON 超大整数在转换 numpy/float 时溢出。
        if any(type(v) not in (int, float) or abs(v) > 1e100 or not math.isfinite(v) for v in p):
            raise ValueError(f'finite numeric {dimensions}D points required')
    if not inside(obj['points'], bounds):
        raise ValueError('path outside local bounds')
    if dimensions == 3 and any(not Z_BOUNDS[0] <= p[2] <= Z_BOUNDS[1] for p in obj['points']):
        raise ValueError('path outside local height bounds')
    return obj
