"""Observed, fixed-height structured BEV for upper-level lateral decisions.

The cross-section gap descriptors and nested free ratios are an adapted
baseline from *Task-Aligned LiDAR Representation for Autonomous Under-Canopy
UAV Navigation in Plantation Forests* (ICRA 2027 review manuscript 1157,
Section III-B). They are not claimed as new features. Unlike its occupancy
complement, this representation distinguishes ray-observed body-free space
from unknown space and retains every raw collision-relevant hit. It describes
one selected drone-center height, not vertical passability or a complete path.

All coordinates are local meters: x forward, y left, z up. Gap widths describe
already-inflated drone-center free space; subtracting the drone diameter again
would count the body envelope twice. They are grid estimates, not continuous
collision certificates. Longitudinal reachability must be checked separately.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import (BOUNDS, GRID_RESOLUTION, INFLATION, VERTICAL_RADIUS,
                       inside_polygon, observed_centers, observed_body_height)
from .observation import grid_observation


GAP_NAMES = ('widest', 'center', 'left', 'right')
PASS_BANDS = (.75, 1.25, 1.75)


@dataclass
class StructuredBEV:
    """Shared local information, with arrays indexed [forward, lateral]."""
    xs: np.ndarray
    ys: np.ndarray
    distance: np.ndarray
    observed: np.ndarray
    ray_observed: np.ndarray
    actual_body_supported: np.ndarray
    blocked: np.ndarray
    unknown: np.ndarray
    free: np.ndarray
    allowed: np.ndarray
    hit_count: np.ndarray
    occupancy_density: np.ndarray
    sections: list
    gap_widths: np.ndarray
    gap_centers: np.ndarray
    gap_edges: np.ndarray
    pass_free: np.ndarray
    pass_observed: np.ndarray
    pass_unknown: np.ndarray
    bounds: tuple
    resolution: float
    height: float
    max_acceleration: float
    pass_bands: tuple
    actual_start: Optional[np.ndarray]
    body_padding: float
    center_clearance: Optional[float] = None

    def spec(self):
        return {
            'name': 'observed_horizontal_structured_bev_v1',
            'bounds': list(self.bounds), 'resolution': self.resolution,
            'height': self.height, 'max_acceleration': self.max_acceleration,
            'axes': {'x': 'forward', 'y': 'left', 'z': 'up'},
            'array_order': ['forward', 'lateral'],
            'horizontal_collision_radius': INFLATION,
            'vertical_collision_radius': VERTICAL_RADIUS,
            'blocked_rule': 'any raw hit within spherical center clearance' if self.center_clearance is not None else 'any raw hit within cylindrical collision envelope',
            'center_clearance_m': self.center_clearance,
            'free_rule': 'whole body source-supported (rays or optional measured actual body), unblocked, inside corridor',
            'unknown_rule': 'inside corridor, unblocked, whole body not observed',
            'width_rule': 'body-center grid-cell span; no second body subtraction',
            'width_quantization_m': self.resolution,
            'gap_order': list(GAP_NAMES),
            'side_gap_rule': 'largest interval portion in positive/negative y',
            'pass_bands_m': list(self.pass_bands),
            'connectivity_rule': 'center gap extends beyond first lateral band in same cross-section',
            'longitudinal_reachability': 'not encoded by cross-section summaries',
            'occupancy_density_role': 'advisory; never suppresses blocked raw hits',
            'attribution': 'paper-inspired fixed-height adaptation of manuscript 1157 Section III-B',
            'actual_start': None if self.actual_start is None else self.actual_start.tolist(),
            'actual_body_support_rule': 'each body sample covered by rays OR measured actual body; not circle clearing',
            'actual_body_padding_m': self.body_padding,
        }


def _cell_edges(centers, limits, resolution):
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([max(limits[0], centers[0]-resolution/2),
                         min(limits[1], centers[0]+resolution/2)])
    middle = (centers[:-1]+centers[1:])/2
    return np.r_[max(limits[0], centers[0]-resolution/2), middle,
                 min(limits[1], centers[-1]+resolution/2)]


def contiguous_intervals(ys, mask, bounds=None, resolution=GRID_RESOLUTION):
    """All maximal true runs, retaining inclusive lateral cell indices.

    Width includes represented cells, with map-edge cells clipped to bounds.
    This discretization is explicit so drawings and teacher share one width.
    """
    ys, mask = np.asarray(ys, dtype=float), np.asarray(mask, dtype=bool)
    if ys.ndim != 1 or mask.shape != ys.shape or not len(ys):
        raise ValueError('ys and mask must be nonempty matching 1D arrays')
    if np.any(np.diff(ys) <= 0):
        raise ValueError('lateral centers must increase')
    limits = ((float(ys[0]-resolution/2), float(ys[-1]+resolution/2))
              if bounds is None else tuple(bounds))
    edges = _cell_edges(ys, limits, resolution)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    intervals = []
    for first, last in zip(starts, ends):
        low, high = float(edges[first]), float(edges[last+1])
        intervals.append(dict(index_start=int(first), index_end=int(last),
                              y_min=low, y_max=high,
                              center=(low+high)/2, width=high-low))
    return intervals


def _widest(intervals):
    if not intervals:
        return None
    # Stable geometric tie break; callers must not use this as a unique action label.
    return min(intervals, key=lambda gap: (-round(gap['width'], 12),
                                         abs(gap['center']), gap['y_min']))


def _side_intervals(intervals, left):
    result = []
    for gap in intervals:
        low = max(0., gap['y_min']) if left else gap['y_min']
        high = gap['y_max'] if left else min(0., gap['y_max'])
        if high-low > 1e-10:
            result.append(dict(gap, y_min=low, y_max=high,
                               center=(low+high)/2, width=high-low))
    return result


def summarize_cross_sections(xs, ys, free, observed, unknown, bounds,
                             resolution=GRID_RESOLUTION, pass_bands=PASS_BANDS):
    """Extract gap summaries from the same masks used by a local teacher."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    free, observed, unknown = map(np.asarray, (free, observed, unknown))
    expected = (len(xs), len(ys))
    if any(array.shape != expected for array in (free, observed, unknown)):
        raise ValueError('all masks must have shape [len(xs), len(ys)]')
    if not pass_bands or any(band <= 0 for band in pass_bands):
        raise ValueError('pass bands must be positive')
    widths = np.zeros((len(xs), len(GAP_NAMES)))
    centers = np.full_like(widths, np.nan)
    edges = np.full((*widths.shape, 2), np.nan)
    sections = []
    for row, x in enumerate(xs):
        intervals = contiguous_intervals(ys, free[row], bounds[2:], resolution)
        center = next((gap for gap in intervals
                       if gap['y_min'] < 0. < gap['y_max']), None)
        selected = dict(widest=_widest(intervals), center=center,
                        left=_widest(_side_intervals(intervals, True)),
                        right=_widest(_side_intervals(intervals, False)))
        section = dict(x=float(x), intervals=intervals, **selected,
                       center_connects_left=bool(center is not None and
                                                 center['y_max'] > pass_bands[0]),
                       center_connects_right=bool(center is not None and
                                                  center['y_min'] < -pass_bands[0]),
                       observed_fraction=float(observed[row].mean()),
                       unknown_fraction=float(unknown[row].mean()))
        sections.append(section)
        for column, name in enumerate(GAP_NAMES):
            gap = selected[name]
            if gap is not None:
                widths[row, column] = gap['width']
                centers[row, column] = gap['center']
                edges[row, column] = gap['y_min'], gap['y_max']
    pass_free = np.zeros((len(xs), len(pass_bands)))
    pass_observed, pass_unknown = np.zeros_like(pass_free), np.zeros_like(pass_free)
    for column, band in enumerate(pass_bands):
        cells = np.abs(ys) <= band+1e-10
        if cells.any():
            pass_free[:, column] = free[:, cells].mean(axis=1)
            pass_observed[:, column] = observed[:, cells].mean(axis=1)
            pass_unknown[:, column] = unknown[:, cells].mean(axis=1)
    return sections, widths, centers, edges, pass_free, pass_observed, pass_unknown


def extract_structured_bev(points, evidence, bounds=BOUNDS,
                           resolution=GRID_RESOLUTION, height=0.,
                           max_acceleration=1., corridor: Optional[np.ndarray]=None,
                           pass_bands=PASS_BANDS, occupancy_threshold=5,
                           actual_start=None, body_padding=.05, center_clearance=None):
    """Extract a fixed-altitude shared representation from causal local scans.

    ``evidence`` accepts an [N,3] array or an existing scipy cKDTree. No
    global scene, denoising or teacher route is used. ``actual_start`` optionally
    adds the measured current body's support, using the executor's sample-wise
    union of ray evidence and actual body. This support is retained separately
    from rays-only coverage, and does not clear any colliding raw hit.
    Mission corridor coordinates, when supplied, are local x/y meters.
    """
    bounds, pass_bands = tuple(map(float, bounds)), tuple(map(float, pass_bands))
    if (len(bounds) != 4 or not np.isfinite(bounds).all() or
            bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
        raise ValueError('bounds must be finite (xmin, xmax, ymin, ymax)')
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution must be positive and finite')
    if not np.isfinite(height) or occupancy_threshold <= 0:
        raise ValueError('height must be finite and occupancy threshold positive')
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError('hit points must be finite')
    xs, ys, distance, observed = grid_observation(
        points, evidence, bounds=bounds, resolution=resolution, height=height,
        max_acceleration=max_acceleration, center_clearance=center_clearance)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    ray_observed = observed.copy()
    if actual_start is not None:
        actual_start = np.asarray(actual_start, dtype=float)
        if actual_start.shape != (3,) or not np.isfinite(actual_start).all():
            raise ValueError('actual start must be three finite coordinates')
        if not np.isfinite(body_padding) or body_padding < 0:
            raise ValueError('body padding must be nonnegative and finite')
        centers = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, height)]
        indices = np.flatnonzero(~observed.ravel())
        observed.ravel()[indices] = observed_centers(
            evidence, centers[indices], actual=actual_start,
            vertical=observed_body_height(max_acceleration), body_padding=body_padding)
    actual_body_supported = observed & ~ray_observed
    allowed = np.ones(xx.shape, dtype=bool)
    if corridor is not None:
        corridor = np.asarray(corridor, dtype=float)
        if corridor.ndim != 2 or corridor.shape[1] != 2 or len(corridor) < 3:
            raise ValueError('corridor must be a convex [N,2] polygon with at least 3 vertices')
        allowed = inside_polygon(np.c_[xx.ravel(), yy.ravel()], corridor).reshape(xx.shape)
    # Unobserved obstacles still block: ray coverage never erases a measured hit.
    blocked = distance <= (INFLATION if center_clearance is None else center_clearance)
    free = observed & ~blocked & allowed
    unknown = ~observed & ~blocked & allowed
    relevant = points[np.abs(points[:, 2]-height) <= VERTICAL_RADIUS]
    hit_count = np.histogram2d(relevant[:, 0], relevant[:, 1],
                              bins=(_cell_edges(xs, bounds[:2], resolution),
                                    _cell_edges(ys, bounds[2:], resolution)))[0].astype(np.int32)
    density = np.minimum(hit_count/float(occupancy_threshold), 1.)
    summary = summarize_cross_sections(xs, ys, free, observed & allowed, unknown,
                                      bounds, resolution, pass_bands)
    return StructuredBEV(xs, ys, distance, observed, ray_observed,
                         actual_body_supported, blocked, unknown, free,
                         allowed, hit_count, density, *summary,
                         bounds, float(resolution), float(height),
                         float(max_acceleration), pass_bands, actual_start,
                         float(body_padding), center_clearance)
