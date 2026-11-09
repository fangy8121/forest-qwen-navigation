"""离线森林的三维体素距离场和 A*；所有距离均使用世界坐标米制。"""
import heapq
import itertools
import numpy as np
from numba import njit
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from .geometry import REFERENCE_CLEARANCE


@njit(cache=True)
def rasterize(triangles, origin, resolution, occupied):
    # 三角形与体素盒的分离轴测试，保留未穿过格心的细枝。
    basis = np.eye(3)
    for triangle in triangles:
        tri = (triangle-origin)/resolution
        edges = np.empty((3, 3))
        for k in range(3):
            edges[k] = tri[(k+1)%3]-tri[k]
        axes = np.empty((13, 3))
        axes[:3] = basis
        axes[3] = np.cross(edges[0], edges[1])
        for k in range(3):
            for j in range(3):
                axes[4+3*k+j] = np.cross(edges[k], basis[j])
        low = np.empty(13)
        high = np.empty(13)
        for k in range(13):
            projection = tri @ axes[k]
            radius = .5*np.abs(axes[k]).sum()
            low[k] = projection.min()-radius-1e-9
            high[k] = projection.max()+radius+1e-9
        lo = np.empty(3, np.int64)
        hi = np.empty(3, np.int64)
        for k in range(3):
            lo[k] = max(0, int(np.ceil(tri[:, k].min()-.5)))
            hi[k] = min(occupied.shape[k]-1, int(np.floor(tri[:, k].max()+.5)))
        for x in range(lo[0], hi[0]+1):
            for y in range(lo[1], hi[1]+1):
                for z in range(lo[2], hi[2]+1):
                    if occupied[x, y, z]:
                        continue
                    hit = True
                    for k in range(13):
                        p = x*axes[k, 0]+y*axes[k, 1]+z*axes[k, 2]
                        if p < low[k] or p > high[k]:
                            hit = False
                            break
                    if hit:
                        occupied[x, y, z] = True


MOVES = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)
MOVES = MOVES[np.any(MOVES, axis=1)]


@njit(cache=True)
def astar(free, penalty, start, goal):
    shape = free.shape
    costs = np.full(shape, np.inf)
    parent = np.full(shape, -1, np.int64)
    costs[start[0], start[1], start[2]] = 0.
    queue = [(0., 0., start[0], start[1], start[2])]
    while queue:
        _, cost, x, y, z = heapq.heappop(queue)
        if cost != costs[x, y, z]:
            continue
        if x == goal[0] and y == goal[1] and z == goal[2]:
            route = [(x, y, z)]
            while parent[x, y, z] >= 0:
                p = parent[x, y, z]
                z = p % shape[2]
                y = (p//shape[2]) % shape[1]
                x = p//(shape[1]*shape[2])
                route.append((x, y, z))
            return route[::-1]
        for move in MOVES:
            a, b, c = x+move[0], y+move[1], z+move[2]
            if not (0 <= a < shape[0] and 0 <= b < shape[1] and 0 <= c < shape[2]):
                continue
            # 检查斜向移动接触的所有体素，禁止从障碍边角穿过。
            clear = True
            for i in range(2 if move[0] else 1):
                for j in range(2 if move[1] else 1):
                    for k in range(2 if move[2] else 1):
                        if not free[x+i*move[0], y+j*move[1], z+k*move[2]]:
                            clear = False
            if not clear:
                continue
            candidate = cost+np.sqrt((move*move).sum())*(1.+penalty[a, b, c])
            if candidate < costs[a, b, c]:
                costs[a, b, c] = candidate
                parent[a, b, c] = (x*shape[1]+y)*shape[2]+z
                heuristic = np.sqrt((a-goal[0])**2+(b-goal[1])**2+(c-goal[2])**2)
                heapq.heappush(queue, (candidate+heuristic, candidate, a, b, c))
    raise ValueError('no connected 3D route')


def gate_crossings(path, gates, require_all=True):
    """检查穿过任务平面的位置；局部平滑检查不要求经过全部平面。"""
    crossings = []
    for gate in gates:
        values = []
        for a, b in zip(path[:-1], path[1:]):
            if (a[0]-gate['x'])*(b[0]-gate['x']) <= 0 and abs(b[0]-a[0]) > 1e-9:
                point = np.asarray(a)+(np.asarray(b)-a)*(gate['x']-a[0])/(b[0]-a[0])
                if ('z_min' in gate and point[2] <= gate['z_min']) or ('z_max' in gate and point[2] >= gate['z_max']):
                    return None
                values.append(float(point[1]))
        if (require_all and not values) or any(not gate['y_min'] < y < gate['y_max'] for y in values):
            return None
        crossings.append(values)
    return crossings


class ESDF:
    def __init__(self, vertices, faces, terrain, bounds, height=2., resolution=.2, clearance=REFERENCE_CLEARANCE):
        self.resolution, self.clearance = resolution, clearance
        self.terrain, self.height, self.bounds = terrain, height, bounds
        self.half_diagonal = np.sqrt(3)*resolution/2
        # 场外障碍也参与距离计算；搜索仍限制在给定 bounds 内。
        xy0 = np.array([bounds[0], bounds[2]])-clearance-2*resolution
        xy1 = np.array([bounds[1], bounds[3]])+clearance+2*resolution
        xs, ys = [np.arange(a, b+resolution, resolution) for a, b in zip(xy0, xy1)]
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        ground = terrain(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        self.origin = np.array([xs[0], ys[0], ground.min()-clearance])
        zs = np.arange(self.origin[2], ground.max()+height+2+clearance, resolution)
        occupied = np.zeros((*xx.shape, len(zs)), dtype=np.bool_)
        tri = vertices[faces].astype(float)
        end = self.origin+(np.array(occupied.shape)-1)*resolution
        selected = ((tri.max(axis=1) >= self.origin-resolution).all(axis=1)
                    & (tri.min(axis=1) <= end+resolution).all(axis=1))
        rasterize(tri[selected], self.origin, resolution, occupied)
        occupied |= zs[None, None, :] <= ground[:, :, None]
        occupied = binary_fill_holes(occupied)
        self.distance = (distance_transform_edt(~occupied)*resolution).astype(np.float32)
        self.distance[occupied] = -(distance_transform_edt(occupied)*resolution)[occupied]
        agl = zs[None, None, :]-ground[:, :, None]
        # 仅精化可能被离散化余量误封的窄缝，保留相同的物理净空。
        import trimesh
        boundary = np.argwhere((self.distance > clearance)
                              & (self.distance <= clearance+2*self.half_diagonal)
                              & (agl > .8+self.half_diagonal) & (agl < height+2-self.half_diagonal)
                              & ((xx >= bounds[0]) & (xx <= bounds[1]) & (yy >= bounds[2]) & (yy <= bounds[3]))[:, :, None])
        local_tri = tri[selected]
        mesh = trimesh.Trimesh(vertices=local_tri.reshape(-1, 3),
                              faces=np.arange(local_tri.size//3).reshape(-1, 3), process=False)
        radius = clearance+3*self.half_diagonal
        for first in range(0, len(boundary), 256):
            cells = boundary[first:first+256]
            points = self.origin+cells*resolution
            nearby = [list(mesh.triangles_tree.intersection(np.r_[p-radius,p+radius])) for p in points]
            owners = np.repeat(np.arange(len(points)), [len(ids) for ids in nearby])
            ids = np.concatenate(nearby).astype(int)
            nearest = trimesh.triangles.closest_point(local_tri[ids], points[owners])
            exact = np.full(len(points), radius)
            np.minimum.at(exact, owners, np.linalg.norm(nearest-points[owners], axis=1))
            self.distance[tuple(cells.T)] = exact+self.half_diagonal
        self.free = (self.distance > clearance+2*self.half_diagonal) & (agl > .8+self.half_diagonal) & (agl < height+2-self.half_diagonal)
        self.free &= ((xx >= bounds[0]) & (xx <= bounds[1]) & (yy >= bounds[2]) & (yy <= bounds[3]))[:, :, None]
        self.penalty = (.05*np.abs(agl-height)+.1/np.maximum(self.distance, resolution)).astype(np.float32)
        self.penalty += (.1*np.abs(yy-(bounds[2]+bounds[3])/2))[:, :, None]

    def query(self, points):
        q = (np.asarray(points)-self.origin)/self.resolution
        base = np.floor(q).astype(int)
        result = np.full(len(q), -np.inf)
        # 八个邻近格心各给出距离下界，取最大值避免最近格心切换造成假碰撞。
        for offset in itertools.product((0, 1), repeat=3):
            cells = base+offset
            valid = ((cells >= 0) & (cells < self.distance.shape)).all(axis=1)
            ij = cells[valid]
            lower = self.distance[tuple(ij.T)]-self.half_diagonal-np.linalg.norm(q[valid]-ij, axis=1)*self.resolution
            result[valid] = np.maximum(result[valid], lower)
        return result

    def clear(self, points):
        agl = points[:, 2]-self.terrain(points)
        padding = .5*np.linalg.norm(np.diff(points, axis=0), axis=1).max(initial=0.)
        return bool((self.query(points) > self.clearance+padding).all()
                    and (agl > .8).all() and (agl < self.height+2).all()
                    and (points[:, 0] >= self.bounds[0]).all() and (points[:, 0] <= self.bounds[1]).all()
                    and (points[:, 1] >= self.bounds[2]).all() and (points[:, 1] <= self.bounds[3]).all())

    def route(self, start, goal, gates):
        free = self.free.copy()
        xs = self.origin[0]+np.arange(self.free.shape[0])*self.resolution
        ys = self.origin[1]+np.arange(self.free.shape[1])*self.resolution
        zs = self.origin[2]+np.arange(self.free.shape[2])*self.resolution
        for gate in gates:
            forbidden = ((np.abs(xs[:, None]-gate['x']) <= self.resolution)
                         & ((ys[None, :] <= gate['y_min']+self.resolution/2)
                            | (ys[None, :] >= gate['y_max']-self.resolution/2)))
            free[forbidden] = False
            if 'z_min' in gate or 'z_max' in gate:
                forbidden_z=(zs <= gate.get('z_min',-np.inf)+self.resolution/2) | (zs >= gate.get('z_max',np.inf)-self.resolution/2)
                for ix in np.flatnonzero(np.abs(xs-gate['x']) <= self.resolution):
                    free[ix][:,forbidden_z] = False
        indices = np.rint((np.array([start, goal])-self.origin)/self.resolution).astype(np.int64)
        if not free[tuple(indices[0])] or not free[tuple(indices[1])]:
            raise ValueError('3D start or goal blocked')
        route = self.origin+np.asarray(astar(free, self.penalty, *indices))*self.resolution
        route = np.vstack([start, route, goal])
        def clear(a, b):
            p = np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/.1))+1))
            if not self.clear(p):
                return False
            return gate_crossings([a, b], gates, require_all=False) is not None
        simple, i = [route[0]], 0
        while i < len(route)-1:
            j = len(route)-1
            while j > i+1 and not clear(route[i], route[j]):
                j -= 1
            if not clear(route[i], route[j]):
                raise ValueError('3D endpoint connection blocked')
            if np.linalg.norm(route[j]-simple[-1]) > 1e-6:
                simple.append(route[j])
            i = j
        return np.asarray(simple)
