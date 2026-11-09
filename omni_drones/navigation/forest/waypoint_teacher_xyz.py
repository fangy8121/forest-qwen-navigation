"""Measured-observation XYZ teacher; truth meshes never enter target generation."""
import json
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates
from scipy.spatial import cKDTree

from .esdf import astar
from .geometry import BOUNDS, Z_BOUNDS, GRID_RESOLUTION
from .geometry import INFLATION, VERTICAL_RADIUS, occupied_centers, center_clearance
from .geometry import parse_path, world_to_local
from .planning import PathSafety, smooth_path
from .teacher import route_answer
from .trajectory import QuinticPath
from .waypoint_teacher import load_observation, dense_polyline

GRID_LOWER = np.array([BOUNDS[0], BOUNDS[2], np.ceil(Z_BOUNDS[0]/GRID_RESOLUTION)*GRID_RESOLUTION])
GRID_UPPER = np.array([BOUNDS[1], BOUNDS[3], np.floor(Z_BOUNDS[1]/GRID_RESOLUTION)*GRID_RESOLUTION])


def observation_safety(obs):
    flight = SimpleNamespace(**json.loads(str(obs['flight_json'])))
    corridor = obs.get('corridor')
    return PathSafety(obs['points'], obs['evidence'], corridor if corridor is not None and len(corridor) else None,
                      flight, float(obs.get('lidar_hz', 10.)))


def signed_distance(free, step=GRID_RESOLUTION):
    padded = np.pad(np.asarray(free, bool), 1, constant_values=False)
    crop = (slice(1, -1),)*free.ndim
    positive = distance_transform_edt(padded)[crop]
    negative = distance_transform_edt(~padded)[crop]
    return np.where(free, positive-.5, -(negative-.5)).astype(np.float32)*step


def observed_grid(obs):
    safety = observation_safety(obs)
    step = GRID_RESOLUTION
    shape = np.rint((GRID_UPPER-GRID_LOWER)/step).astype(int)+1
    cells = np.indices(shape).reshape(3, -1).T
    centers = GRID_LOWER+cells*step
    reference = center_clearance(safety.flight) is not None
    free = safety.clear_points(centers, padding=0. if reference else .05)
    free[free] &= safety.known_points(centers[free], np.zeros(3))
    if not reference:
        search = free & ~occupied_centers(safety.obstacles, centers,
                                          INFLATION+step/np.sqrt(2), VERTICAL_RADIUS+step/2)
    else:
        # Nodes are exact centers, not certificates for whole voxel boxes.
        # Every returned edge/curve is densely rechecked with its sampling margin.
        search = free.copy()
    return dict(free=free.reshape(shape), search=search.reshape(shape),
                sdf=signed_distance(free.reshape(shape)), origin=GRID_LOWER.copy(), step=np.float64(step)), safety


def sample_sdf(grid, points):
    indices = ((np.asarray(points)-grid['origin'])/float(grid['step'])).T
    return map_coordinates(grid['sdf'], indices, order=1, mode='constant', cval=-1.)


def assess_answer(answer, obs, grid=None):
    if isinstance(answer, dict):
        answer = json.dumps(answer, allow_nan=False)
    try:
        answer = parse_path(answer)
    except ValueError as error:
        return dict(accepted=False, reason=str(error), status='INVALID')
    if answer['status'] == 'HOLD':
        return dict(accepted=False, reason='hold', status='HOLD', progress_m=0.)
    six = np.asarray(answer['points'])
    safety = observation_safety(obs)
    try:
        curve = safety.reconstruct(np.vstack([np.zeros(3), six]), obs['velocity'], obs['acceleration'], np.zeros(3))
    except ValueError:
        return dict(accepted=False, reason='trajectory_rejected', status='GO',
                    path_checks=safety.checks(dense_polyline(np.vstack([np.zeros(3), six]), .08), np.zeros(3)))
    braking_checks = safety.checks(safety.braking_corridor(curve, 0., np.zeros(3)), np.zeros(3))
    accepted = all(braking_checks.values())
    progress = float(np.linalg.norm(obs['goal'])-np.linalg.norm(obs['goal']-six[-1]))
    result = dict(accepted=accepted, reason='accepted' if accepted else 'braking_rejected',
                  status='GO', progress_m=progress, braking_checks=braking_checks)
    if grid is not None:
        distance = sample_sdf(grid, curve.remaining(0., np.zeros(3)))
        result.update(min_grid_clearance_m=float(distance.min()), mean_grid_clearance_m=float(distance.mean()))
    return result


def centered_teacher(obs, grid=None, safety=None, reference=None):
    if grid is None or safety is None:
        grid, safety = observed_grid(obs)
    review = dict(status='REVIEW', points=[])
    goal = np.asarray(obs['goal'], dtype=float)
    brake = QuinticPath.brake(np.zeros(3), obs['velocity'], obs['acceleration'],
                             safety.flight.max_acceleration, safety.flight.max_jerk)
    braking = brake.remaining(0., np.zeros(3))
    initial_checks = safety.checks(braking, np.zeros(3))
    if np.linalg.norm(goal) < .3:
        if all(initial_checks.values()) and np.linalg.norm(braking[-1]-goal) < .3:
            return dict(status='HOLD', points=[]), dict(reason='goal_reached', accepted=True,
                                                      initial_braking=initial_checks)
        return review, dict(reason='goal_stop_unverified', accepted=False, initial_braking=initial_checks)
    free = grid['search']
    start = np.rint(-grid['origin']/float(grid['step'])).astype(np.int64)
    if not free[tuple(start)]:
        return review, dict(reason='start_not_observed_free', accepted=False, initial_braking=initial_checks)
    # One unverifiable emergency-stop template does not rule out other controls.
    # Every candidate still passes reconstruction and its own braking guard.
    connected, _ = label(free)
    reachable = connected == connected[tuple(start)]
    cells = np.indices(free.shape).reshape(3, -1).T
    centers = grid['origin']+cells*float(grid['step'])
    direction = goal/np.linalg.norm(goal)
    candidates = np.flatnonzero(reachable.ravel() & (centers @ direction >= .5))
    score = np.linalg.norm(centers[candidates]-goal, axis=1)+.1*np.linalg.norm(centers[candidates], axis=1)
    penalty = (1.5*np.exp(-np.maximum(grid['sdf'], 0.)/.35)
               + .03*np.abs(centers[:, 2]-goal[2]).reshape(free.shape)).astype(np.float32)
    if reference is not None and len(reference) > 1:
        # A soft preference; collision/evidence constraints are never changed.
        penalty += .15*cKDTree(dense_polyline(reference, .1)).query(centers)[0].reshape(free.shape)
    tried, reason = [], 'no_reachable_subgoal'
    for index in candidates[np.argsort(score, kind='stable')]:
        target = centers[index]
        if tried and min(np.linalg.norm(target-p) for p in tried) < .8:
            continue
        tried.append(target)
        route_cells = np.asarray(astar(free, penalty, start, cells[index]))
        route = grid['origin']+route_cells*float(grid['step'])
        floor = max(.025, min(.18, float(sample_sdf(grid, route).min())-.04))
        def margin_ok(points):
            return bool((sample_sdf(grid, dense_polyline(points)) >= floor).all())
        def safe(points):
            dense = dense_polyline(points, .08)
            return safety.clear(dense) and safety.known(dense, np.zeros(3)) and margin_ok(dense)
        answer, reason = route_answer(route, safe, safety, safety.flight,
                                      obs['velocity'], obs['acceleration'], shortcut_check=margin_ok)
        if answer is not None:
            audit = assess_answer(answer, obs, grid)
            if audit['accepted'] and audit['progress_m'] >= .3:
                return answer, dict(audit, reason='centered_executable_xyz_path', clearance_floor_m=floor,
                                    targets_tried=len(tried), initial_braking=initial_checks)
            reason = 'insufficient_goal_progress' if audit['accepted'] else audit['reason']
        if len(tried) >= 9:
            break
    return review, dict(reason=reason, accepted=False, targets_tried=len(tried), initial_braking=initial_checks)


def truncate_reference(points, horizon):
    points = np.asarray(points, dtype=float)
    distance = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    unique = np.r_[True, np.diff(distance) > 1e-8]
    points, distance = points[unique], distance[unique]
    end = min(float(distance[-1]), horizon)
    query = np.r_[distance[distance < end], end]
    return np.column_stack([np.interp(query, distance, points[:, axis]) for axis in range(3)])


def reference_prefix(curve, timestamp, position, yaw, settle_seconds=3., horizon=6.):
    """Teacher-only future reference, aligned with flight.run_flight's clock."""
    elapsed = np.clip(timestamp-settle_seconds, 0., curve.duration)
    times = np.r_[np.arange(elapsed, curve.duration, .05), curve.duration]
    local = world_to_local(curve.sample(times)[0], position, yaw)
    return truncate_reference(np.vstack([np.zeros(3), local]), horizon)


def reference_teacher(obs, reference, grid=None, safety=None):
    """Original demonstration first, bounded local repair second, never fake HOLD."""
    from .teacher import six_point_candidates
    if grid is None or safety is None:
        grid, safety = observed_grid(obs)
    attempts = []
    if np.linalg.norm(obs['goal']) >= .3:
        previous_length = None
        for horizon in (6., 4., 2.):
            segment = truncate_reference(reference, horizon)
            length = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
            if length < .3 or (previous_length is not None and abs(length-previous_length) < .01):
                continue
            previous_length = length
            for six in six_point_candidates(segment):
                answer = dict(status='GO', points=np.round(six, 4).tolist())
                audit = assess_answer(answer, obs, grid)
                attempts.append(dict(horizon_m=length, **audit))
                if audit['accepted'] and audit['progress_m'] >= .3:
                    return answer, dict(audit, reason='reference_executable_xyz_path',
                                        source='reference', reference_attempts=attempts)
    answer, audit = centered_teacher(obs, grid, safety, reference=reference)
    return answer, dict(audit, source='local_repair' if answer['status']=='GO' else 'unresolved' if answer['status']=='REVIEW' else 'verified_stop',
                        reference_attempts=attempts)
