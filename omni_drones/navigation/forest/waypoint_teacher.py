"""Observed-only horizontal teacher and geometric targets for Qwen waypoints.

No simulator mesh, reference flight or global map is used here. The signed
distance is to *any forbidden center*: collision, unknown, or outside corridor.
It is a training surrogate; continuous trajectory/braking checks remain required.
"""
import json
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates

from .esdf import astar
from .geometry import BOUNDS, GRID_RESOLUTION, parse_path
from .planning import PathSafety, smooth_path
from .teacher import route_answer


def load_observation(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def observation_safety(obs):
    flight = SimpleNamespace(**json.loads(str(obs['flight_json'])))
    corridor = obs.get('corridor')
    if corridor is not None and not len(corridor):
        corridor = None
    return PathSafety(obs['points'], obs['evidence'], corridor, flight,
                      float(obs.get('lidar_hz', 10.)), height_bounds=(-.05, .05))


def signed_distance(free, step=GRID_RESOLUTION):
    """Cell-centered signed distance, with a forbidden border outside the grid."""
    padded = np.pad(np.asarray(free, bool), 1, constant_values=False)
    positive = distance_transform_edt(padded)[1:-1, 1:-1]
    negative = distance_transform_edt(~padded)[1:-1, 1:-1]
    return np.where(free, positive-.5, -(negative-.5)).astype(np.float32)*step


def observed_grid(obs):
    safety = observation_safety(obs)
    step = GRID_RESOLUTION
    xs = np.linspace(BOUNDS[0], BOUNDS[1], round((BOUNDS[1]-BOUNDS[0])/step)+1)
    ys = np.linspace(BOUNDS[2], BOUNDS[3], round((BOUNDS[3]-BOUNDS[2])/step)+1)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    centers = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    # Match exact executor padding. Search separately adds cell-diagonal padding.
    free = safety.clear_points(centers, padding=.05)
    free[free] &= safety.known_points(centers[free], np.zeros(3))
    search = free & safety.clear_points(centers, padding=step/np.sqrt(2))
    shape = xx.shape
    return dict(free=free.reshape(shape), search=search.reshape(shape),
                sdf=signed_distance(free.reshape(shape)),
                origin=np.array([xs[0], ys[0]], np.float32), step=np.float32(step)), safety


def sample_sdf(grid, points):
    xy = np.asarray(points)[:, :2]
    indices = ((xy-grid['origin'])/float(grid['step'])).T
    return map_coordinates(grid['sdf'], indices, order=1, mode='constant', cval=-1.)


def dense_polyline(points, spacing=.05):
    points = np.asarray(points)
    if len(points) < 2:
        return points.copy()
    return np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/spacing))+1))
                           for a, b in zip(points[:-1], points[1:])])


def assess_answer(answer, obs, grid=None):
    """Actual smooth-trajectory/start-braking validation, never repairs predictions."""
    if isinstance(answer, dict):
        answer = json.dumps(answer, allow_nan=False)
    try:
        answer = parse_path(answer)
    except ValueError as error:
        return dict(accepted=False, reason=str(error), status='INVALID')
    if answer['status'] == 'HOLD':
        return dict(accepted=False, reason='hold', status='HOLD', progress_m=0.)
    six = np.asarray(answer['points'])
    if np.max(np.abs(six[:, 2])) > 1e-6:
        return dict(accepted=False, reason='nonhorizontal', status='GO')
    safety = observation_safety(obs)
    def safe(points):
        return safety.clear(points) and safety.known(dense_polyline(points, .08), np.zeros(3))
    p = np.vstack([np.zeros(3), six])
    try:
        curve = smooth_path(p, safety.flight, safe, obs['velocity'], obs['acceleration'])
    except ValueError:
        return dict(accepted=False, reason='trajectory_rejected', status='GO')
    accepted = bool(safety.guard(curve, 0., np.zeros(3)))
    progress = float(np.linalg.norm(obs['goal'][:2])-np.linalg.norm(obs['goal'][:2]-six[-1, :2]))
    result = dict(accepted=accepted, reason='accepted' if accepted else 'braking_rejected',
                  status='GO', progress_m=progress)
    if grid is not None:
        distance = sample_sdf(grid, curve.remaining(0., np.zeros(3)))
        result.update(min_grid_clearance_m=float(distance.min()),
                      mean_grid_clearance_m=float(distance.mean()))
    return result


def centered_teacher(obs, grid=None, safety=None):
    """Deterministic clearance-weighted A*, then validated six-point reconstruction.

    A canonical teacher route is kept even when another route might be feasible.
    Unsuccessful search is labelled HOLD, not a proof that no route exists.
    """
    if grid is None or safety is None:
        grid, safety = observed_grid(obs)
    hold = dict(status='HOLD', points=[])
    goal = np.array(obs['goal'], dtype=float).copy(); goal[2] = 0.
    if np.linalg.norm(goal[:2]) < .3:
        return hold, dict(reason='horizontal_goal_reached')
    free = grid['search']
    start = np.rint(-grid['origin']/float(grid['step'])).astype(np.int64)
    if not free[tuple(start)]:
        return hold, dict(reason='start_not_observed_free')
    connected, _ = label(free)
    reachable = connected == connected[tuple(start)]
    cells = np.indices(free.shape).reshape(2, -1).T
    xy = grid['origin']+cells*float(grid['step'])
    direction = goal[:2]/np.linalg.norm(goal[:2])
    candidates = np.flatnonzero(reachable.ravel() & (xy @ direction >= .5))
    score = np.linalg.norm(xy[candidates]-goal[:2], axis=1)+.1*np.linalg.norm(xy[candidates], axis=1)
    # Penalize proximity to both physical and observation/corridor boundaries.
    penalty = (1.5*np.exp(-np.maximum(grid['sdf'], 0.)/.35)).astype(np.float32)
    tried, reason = [], 'no_reachable_subgoal'
    for index in candidates[np.argsort(score, kind='stable')]:
        target = xy[index]
        if tried and min(np.linalg.norm(target-p) for p in tried) < .8:
            continue
        tried.append(target)
        route_cells = np.asarray(astar(free[:, :, None], penalty[:, :, None],
                                      np.r_[start, 0], np.r_[cells[index], 0]))
        route = np.c_[grid['origin']+route_cells[:, :2]*float(grid['step']),
                      np.zeros(len(route_cells))]
        # Preserve the clearance found by A*: don't shortcut back to obstacle edges.
        floor = max(.025, min(.18, float(sample_sdf(grid, route).min())-.04))
        def margin_ok(points):
            return bool((sample_sdf(grid, dense_polyline(points)) >= floor).all())
        def safe(points):
            dense = dense_polyline(points, .08)
            return (safety.clear(dense) and safety.known(dense, np.zeros(3))
                    and margin_ok(dense))
        answer, reason = route_answer(route, safe, safety, safety.flight,
                                      obs['velocity'], obs['acceleration'], shortcut_check=margin_ok)
        if answer is not None:
            # Teacher projection only; revalidate with measured initial v/a.
            for point in answer['points']:
                point[2] = 0.
            audit = assess_answer(answer, obs, grid)
            if audit['accepted'] and audit['progress_m'] >= .3:
                return answer, dict(audit, reason='centered_executable_path', clearance_floor_m=floor,
                                    targets_tried=len(tried))
        if len(tried) >= 9:
            break
    return hold, dict(reason=reason, targets_tried=len(tried))
