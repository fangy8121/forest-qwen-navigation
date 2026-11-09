"""Reference search in the online planner's own geometry.

``ESDF`` admits a voxel when a sphere around it is clear and its AGL lies in
0.8..height+2 m. The online stack instead flies ``desired_agl`` +- a small band
and checks a vertical cylinder: horizontal clearance to every surface within
+-``vertical_m`` of the body. References found by the sphere search were vetoed
at the end by that cylinder (14 of 18 rejected maps) or flew heights the online
mission never commands (run 46677b1e: a 3.7 m climb over a low-branch layer).
"""
import itertools

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter1d

from omni_drones.navigation.forest.esdf import ESDF


class NavigationESDF(ESDF):
    def __init__(self, vertices, faces, terrain, bounds, *, height, resolution, horizontal, vertical,
                 agl_band, agl_check=None):
        low, high = map(float, agl_band)
        if not 0 < low < height < high:
            raise ValueError('agl_band must bracket the nominal height')
        # Search inside ``agl_band``; continuous curves may use the slightly
        # wider ``agl_check`` (corner cutting over terrain changes).
        self.agl_check = (low, high) if agl_check is None else tuple(map(float, agl_check))
        if not self.agl_check[0] <= low < high <= self.agl_check[1]:
            raise ValueError('agl_check must contain agl_band')
        # The sphere clearance is only used by the parent to size the grid.
        super().__init__(vertices, faces, terrain, bounds, height=height, resolution=resolution,
                         clearance=horizontal)
        self.horizontal, self.vertical, self.agl_band = float(horizontal), float(vertical), (low, high)
        occupied = self.distance <= 0
        # One extra cell covers queries between two cell centres.
        reach = int(np.ceil(self.vertical/resolution))+1
        band = maximum_filter1d(occupied, size=2*reach+1, axis=2, mode='nearest')
        # Per-layer 2-D distance: a huge z sampling keeps layers independent.
        self.horizontal_distance = (distance_transform_edt(~band, sampling=(resolution, resolution, 1e6))
                                    .astype(np.float32))
        self.half_diagonal_xy = np.sqrt(2)*resolution/2
        shape = self.distance.shape
        xs = self.origin[0]+np.arange(shape[0])*resolution
        ys = self.origin[1]+np.arange(shape[1])*resolution
        zs = self.origin[2]+np.arange(shape[2])*resolution
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        ground = terrain(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        agl = zs[None, None, :]-ground[:, :, None]
        # Coarse admission only; every simplified segment is re-checked by
        # ``clear`` with exact lower bounds, and the map by the mesh veto.
        # Margin = the lower bound ``clear`` applies halfway between two free
        # cells (half diagonal + half a cell) plus its 0.1 m sampling padding.
        self.free = ((self.horizontal_distance > self.horizontal+self.half_diagonal_xy+resolution/2+.05)
                     & (agl > low+resolution/2) & (agl < high-resolution/2))
        self.free &= ((xx >= bounds[0]) & (xx <= bounds[1]) & (yy >= bounds[2]) & (yy <= bounds[3]))[:, :, None]
        # Prefer the nominal height strongly: the online mission holds it.
        self.penalty = (.5*np.abs(agl-height)
                        + .1/np.maximum(self.horizontal_distance, resolution)).astype(np.float32)
        self.penalty += (.1*np.abs(yy-(bounds[2]+bounds[3])/2))[:, :, None]

    def query_horizontal(self, points):
        q = (np.asarray(points)-self.origin)/self.resolution
        base = np.floor(q).astype(int)
        result = np.full(len(q), -np.inf)
        for offset in itertools.product((0, 1), repeat=3):
            cells = base+offset
            valid = ((cells >= 0) & (cells < self.horizontal_distance.shape)).all(axis=1)
            ij = cells[valid]
            lower = (self.horizontal_distance[tuple(ij.T)]-self.half_diagonal_xy
                     - np.linalg.norm(q[valid, :2]-ij[:, :2], axis=1)*self.resolution)
            result[valid] = np.maximum(result[valid], lower)
        return result

    def clear(self, points):
        points = np.asarray(points, float)
        agl = points[:, 2]-self.terrain(points)
        padding = .5*np.linalg.norm(np.diff(points, axis=0), axis=1).max(initial=0.)
        low, high = self.agl_check
        return bool((self.query_horizontal(points) > self.horizontal+padding).all()
                    and (agl > low).all() and (agl < high).all()
                    and (points[:, 0] >= self.bounds[0]).all() and (points[:, 0] <= self.bounds[1]).all()
                    and (points[:, 1] >= self.bounds[2]).all() and (points[:, 1] <= self.bounds[3]).all())
