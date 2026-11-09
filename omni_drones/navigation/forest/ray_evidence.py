"""Exact positive-length voxel traversal; never extends a ray beyond its hit."""
import numpy as np
from numba import njit


@njit(cache=True)
def trace_free_voxels(seen, origin, directions, distances, lower, step):
    bounds_low = lower-step/2
    shape = np.array(seen.shape)
    bounds_high = bounds_low+shape*step
    for ray in range(len(directions)):
        direction = directions[ray]
        enter, end = 0., distances[ray]
        if not np.isfinite(end) or end <= 0:
            continue
        for axis in range(3):
            d = direction[axis]
            if abs(d) < 1e-12:
                if origin[axis] < bounds_low[axis] or origin[axis] >= bounds_high[axis]:
                    end = -1.
            else:
                a = (bounds_low[axis]-origin[axis])/d
                b = (bounds_high[axis]-origin[axis])/d
                enter = max(enter, min(a, b))
                end = min(end, max(a, b))
        if end-enter <= 1e-9:
            continue
        cell = np.floor((origin+direction*(enter+1e-9)-bounds_low)/step).astype(np.int64)
        stride = np.sign(direction).astype(np.int64)
        next_t = np.full(3, np.inf)
        delta_t = np.full(3, np.inf)
        for axis in range(3):
            if abs(direction[axis]) > 1e-12:
                edge = bounds_low[axis]+(cell[axis]+(stride[axis] > 0))*step
                next_t[axis] = (edge-origin[axis])/direction[axis]
                delta_t[axis] = step/abs(direction[axis])
        while enter < end-1e-9 and (cell >= 0).all() and (cell < shape).all():
            leave = min(end, next_t.min())
            if leave-enter > 1e-9:
                seen[cell[0], cell[1], cell[2]] = True
            if leave >= end-1e-9:
                break
            for axis in range(3):
                if next_t[axis] <= leave+1e-9:
                    cell[axis] += stride[axis]
                    next_t[axis] += delta_t[axis]
            enter = leave
