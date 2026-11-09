"""Supported terrain from measured returns, independent of scene metadata."""
from functools import lru_cache

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

from .terrain_following import GroundSurface, supported_triangles


@lru_cache(maxsize=4)
def _triangulate(vertices_bytes, count, max_edge, max_slope):
    vertices = np.frombuffer(vertices_bytes, dtype=np.float64).reshape(count, 3)
    try:
        tri = Delaunay(vertices[:, :2])
    except QhullError:
        return None, None, None
    valid = supported_triangles(vertices, tri, max_edge, max_slope)
    return tri, valid, cKDTree(vertices[:, :2])


def query_terrain(terrain, xy, now=None, max_age=None):
    """Interpolate only inside supported triangles; unknown heights stay NaN.

    Historical terrain may guide return intentions in a static scene. Passing
    ``max_age`` additionally requires every supporting vertex to be recent;
    neither mode replaces the lower executor's fresh 3-D obstacle checks.
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    z, valid = np.full(len(xy), np.nan), np.zeros(len(xy), bool)
    if not terrain:
        return z, valid
    index = terrain.get('_index')
    if index is not None:
        # Built once per terrain version by ObservedTerrain.snapshot (in the
        # structure worker); the control thread only interpolates.
        vertices, times, (tri, supported, tree) = index['vertices'], index['timestamps'], index['triangulation']
    else:
        vertices = np.asarray(terrain.get('vertices', []), np.float64).reshape(-1, 3)
        times = np.asarray(terrain.get('timestamps', []), float)
    if (len(vertices) < 3 or len(times) != len(vertices)
            or not np.isfinite(vertices).all() or not np.isfinite(times).all()):
        return z, valid
    if index is None:
        tri, supported, tree = _triangulate(vertices.tobytes(), len(vertices),
            float(terrain.get('max_edge', 1.8)), float(terrain.get('max_slope', .7)))
    if tri is None:
        return z, valid
    sid = np.full(len(xy), -1, int)
    finite = np.isfinite(xy).all(axis=1)
    sid[finite] = tri.find_simplex(xy[finite])
    ids = np.flatnonzero(sid >= 0)
    ids = ids[supported[sid[ids]]]
    if now is not None:
        ages = float(now)-times[tri.simplices[sid[ids]]]
        recent = (ages >= -1e-6).all(axis=1)
        if max_age is not None:
            recent &= (ages <= float(max_age)).all(axis=1)
        ids = ids[recent]
    elif max_age is not None:
        raise ValueError('now is required with max_age')
    simplex = sid[ids]
    uv = np.einsum('nij,nj->ni', tri.transform[simplex, :2],
                   xy[ids]-tri.transform[simplex, 2])
    weights = np.c_[uv, 1-uv.sum(axis=1)]
    z[ids] = (weights*vertices[tri.simplices[simplex], 2]).sum(axis=1)
    valid[ids] = True
    # MID360's limited downward FOV leaves a small hole immediately under the
    # body. Interpolate it only when nearby ground brackets the query in angle
    # and agrees with one local plane; an open edge or large hole stays unknown.
    fallback = np.flatnonzero(finite & ~valid & (sid >= 0))
    if len(fallback):
        distances, neighbors = tree.query(xy[fallback], k=min(64, len(vertices)))
        radius = float(terrain.get('bracket_radius', 3.2))
        for index, distance, neighbor in zip(fallback, distances, neighbors):
            keep = distance <= radius
            neighbor = neighbor[keep]
            if now is not None:
                age = float(now)-times[neighbor]
                recent = age >= -1e-6
                if max_age is not None:
                    recent &= age <= float(max_age)
                neighbor = neighbor[recent]
            if len(neighbor) < 10:
                continue
            # Start locally on curved terrain. A fixed large stencil can fail
            # residual checks because distant ground bends, although nearer
            # measured points already bracket and support the query.
            for count in sorted({min(16, len(neighbor)), min(32, len(neighbor)), len(neighbor)}):
                points = vertices[neighbor[:count]]
                delta = points[:, :2]-xy[index]
                angles = np.sort(np.arctan2(delta[:, 1], delta[:, 0]))
                if np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])) >= .95*np.pi:
                    continue
                design = np.c_[delta, np.ones(len(delta))]
                plane, _, rank, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)
                error = np.abs(design@plane-points[:, 2])
                if (rank == 3 and np.linalg.norm(plane[:2]) <= float(terrain.get('max_slope', .7))
                        and error.max() <= .30 and np.sqrt(np.mean(error**2)) <= .15):
                    z[index], valid[index] = plane[2], True
                    break
    return z, valid


class ObservedTerrain:
    """A rolling raw-return window and compact persistent ground vertices.

    Raw returns are capped spatially; old fitted terrain is retained for task
    memory. Reobservations replace estimates, and each vertex keeps its source
    time. An explicit storage failure is preferable to silently losing a region.
    """
    def __init__(self, *, cell=.6, radius=24., max_points=180000,
                 max_vertices=40000, update_seconds=1.):
        self.cell, self.radius = float(cell), float(radius)
        self.max_points, self.max_vertices = max_points, max_vertices
        self.update_seconds, self.next_update = float(update_seconds), -np.inf
        self.points, self.ground = {}, {}
        self.version, self.timestamp = 0, 0.
        self._snapshot = None

    def add(self, points_world, now, position):
        points = np.asarray(points_world, float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        cells = np.unique(np.rint(points/.2).astype(np.int32), axis=0)
        self.points.update((tuple(c), float(now)) for c in cells)
        if now < self.next_update:
            return
        position = np.asarray(position)
        keys = np.asarray(list(self.points), float).reshape(-1, 3)
        keep = np.linalg.norm(keys[:, :2]*.2-position[:2], axis=1) <= self.radius
        self.points = {tuple(k.astype(int)):self.points[tuple(k.astype(int))] for k in keys[keep]}
        if len(self.points) > self.max_points:
            raise RuntimeError('TERRAIN_RAW_STORAGE_BUDGET')
        keys = np.asarray(list(self.points), float).reshape(-1, 3)
        if len(keys):
            times = np.fromiter(self.points.values(), float)
            surface = GroundSurface(keys*.2, times, now, cell=self.cell,
                max_edge=1.8, max_slope=.7, max_age=30.)
            if len(surface.vertices):
                for vertex, age in zip(surface.vertices, surface.ages):
                    key = tuple(np.floor(vertex[:2]/self.cell).astype(int))
                    self.ground[key] = (vertex.tolist(), float(now-age))
        if len(self.ground) > self.max_vertices:
            raise RuntimeError('TERRAIN_VERTEX_STORAGE_BUDGET')
        self.version += 1
        self.timestamp = float(now)
        self._snapshot = None
        self.next_update = float(now)+self.update_seconds

    def snapshot(self):
        if self._snapshot is None:
            values = list(self.ground.values())
            self._snapshot = dict(schema='observed_supported_terrain_v1', version=self.version,
                timestamp=self.timestamp, vertices=tuple(tuple(v) for v, _ in values),
                timestamps=tuple(t for _, t in values), max_edge=1.8, max_slope=.7, bracket_radius=3.2,
                semantics='measured lower surface; no extrapolation or clearance claim')
            # Arrays and triangulation for query_terrain, once per version. The
            # control thread used to rebuild both on every guidance lift; with
            # the whole explored ground that grew to ~0.06 s per goal activation
            # (run c49e9da0). Underscore keys are never written to event logs.
            vertices = np.asarray(self._snapshot['vertices'], np.float64).reshape(-1, 3)
            times = np.asarray(self._snapshot['timestamps'], float)
            triangulation = ((None, None, None) if len(vertices) < 3 else
                _triangulate(vertices.tobytes(), len(vertices), 1.8, .7))
            self._snapshot['_index'] = dict(vertices=vertices, timestamps=times, triangulation=triangulation)
        # Metadata may be replaced by a caller; the large immutable vertex/time
        # tuples are shared until the next actual terrain update.
        return dict(self._snapshot)
