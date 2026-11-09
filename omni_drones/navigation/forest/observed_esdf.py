"""Local distance/gradient queries from online hits, with separate ray evidence.

This field never reads a scene mesh. Distance means distance to *observed* hits;
even an empty, large-distance cell can be unknown. The field is an optimization
aid, not a replacement for PathSafety's body and braking checks.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from .geometry import GRID_RESOLUTION


@dataclass
class DistanceQuery:
    distance: np.ndarray
    gradient: np.ndarray
    known: np.ndarray
    in_bounds: np.ndarray


def _xyz(value):
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, 3))
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError('Expected finite world XYZ points with shape (N, 3)')
    return array


class ObservedESDF:
    """Dense local EDT with analytic trilinear gradients and a known-cell mask.

    A conservative voxel/interpolation correction is subtracted from the EDT:
    half a voxel diagonal for hit quantization and half for interpolation.
    Positive values lower-bound distance to the supplied hit points, within
    the grid. They do not bound distance to unobserved surfaces. Negative values
    near occupied cells are a soft penalty, not a reconstructed solid interior.
    """
    schema = 'observed_local_edt_v1'

    def __init__(self, points, evidence, resolution=GRID_RESOLUTION, *, bounds=None,
                 max_voxels=2_000_000):
        points, evidence = _xyz(points), _xyz(evidence)
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('Resolution must be finite and positive')
        self.resolution = float(resolution)
        arrays = [a for a in (points, evidence) if len(a)]
        if bounds is not None:
            bounds = _xyz(bounds)
            if bounds.shape != (2, 3) or (bounds[1] <= bounds[0]).any():
                raise ValueError('Bounds must contain ordered lower/upper XYZ corners')
            arrays.append(bounds)
        if not arrays:
            raise ValueError('Empty observations require explicit local bounds')
        all_points = np.concatenate(arrays)
        self.origin = (np.floor(all_points.min(axis=0)/resolution)-2)*resolution
        upper = (np.ceil(all_points.max(axis=0)/resolution)+2)*resolution
        shape = np.rint((upper-self.origin)/resolution).astype(np.int64)+1
        if int(np.prod(shape.astype(object))) > max_voxels:
            raise ValueError('Observed ESDF exceeds the local voxel budget')
        self.shape = tuple(shape)
        self.upper = self.origin+(shape-1)*resolution
        self.occupied = np.zeros(self.shape, dtype=bool)
        self.known = np.zeros(self.shape, dtype=bool)
        if len(points):
            self.occupied[tuple(self._cells(points).T)] = True
        if len(evidence):
            self.known[tuple(self._cells(evidence).T)] = True
        self.known[self.occupied] = False
        # EDT on an all-empty grid has implicit-boundary semantics in SciPy;
        # handle it explicitly instead of creating a fictitious obstacle.
        self.has_obstacles = bool(len(points))
        self.empty_distance = float(np.linalg.norm(upper-self.origin))
        self.distances = (distance_transform_edt(~self.occupied, sampling=resolution)
                          - np.sqrt(3)*resolution if len(points)
                          else np.full(self.shape, self.empty_distance))
        self.evidence = cKDTree(evidence)

    def _cells(self, points):
        return np.rint((points-self.origin)/self.resolution).astype(np.int64)

    def query(self, points):
        points = _xyz(points)
        q = (points-self.origin)/self.resolution
        inside = ((q >= 0.) & (q <= np.asarray(self.shape)-1)).all(axis=1)
        q = np.clip(q, 0., np.asarray(self.shape)-1)
        cell = np.minimum(np.floor(q).astype(np.int64), np.asarray(self.shape)-2)
        fraction = q-cell
        distance, gradient = np.zeros(len(q)), np.zeros((len(q), 3))
        for offset in product((0, 1), repeat=3):
            offset = np.asarray(offset)
            weights = np.where(offset, fraction, 1.-fraction)
            values = self.distances[tuple((cell+offset).T)]
            distance += values*weights.prod(axis=1)
            for axis in range(3):
                other = [d for d in range(3) if d != axis]
                gradient[:, axis] += (values*(2*offset[axis]-1)
                                      *weights[:, other].prod(axis=1)/self.resolution)
        nearest = np.rint(q).astype(np.int64)
        known = inside & self.known[tuple(nearest.T)]
        # Out-of-bounds queries cannot acquire free-space semantics by clipping.
        distance[~inside], gradient[~inside] = 0., 0.
        return DistanceQuery(distance, gradient, known, inside)

    def evidence_distance(self, points):
        """Soft center-to-evidence penalty only; does not certify body coverage."""
        points = _xyz(points)
        if not self.evidence.n:
            return np.full(len(points), self.empty_distance), np.zeros_like(points)
        distance, indices = self.evidence.query(points)
        gradient = (points-self.evidence.data[indices])/np.maximum(distance[:, None], 1e-12)
        return distance, gradient

    def save(self, path):
        np.savez_compressed(path, schema=self.schema, origin=self.origin,
                            resolution=self.resolution, distance=self.distances,
                            known=self.known, occupied=self.occupied,
                            correction_m=np.sqrt(3)*self.resolution,
                            semantics='distance to observed hits; known is independent')
