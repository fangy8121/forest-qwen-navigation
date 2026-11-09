"""离线评测：三维轨迹、独立真值网格与执行筛选分别统计。"""
import json
from types import SimpleNamespace
import numpy as np
from .planning import PathSafety, smooth_path
from .teacher import DEFAULT_FLIGHT
from .geometry import (parse_path, local_to_world, inside_polygon, RADIUS, INFLATION,
                       BODY_VERTICAL_RADIUS, VERTICAL_RADIUS, PROTOCOL_VERSION)
from .mesh import horizontal_section, projected_clearance
from .geometry import center_clearance
from .trajectory import QuinticPath


def sample_polyline(path, spacing=.04):
    return np.concatenate([path[:1]] +
        [np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/spacing))+1))[1:]
         for a, b in zip(path[:-1], path[1:])])


def mesh_collision(path, truth, radius, vertical, padding=.025):
    """独立网格检查：沿三维曲线扫掠圆柱包络，采样余量覆盖相邻采样间隙。"""
    if '_triangle_tree' not in truth:
        import trimesh
        vertices, faces = truth['mesh']
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        truth['_triangle_tree'] = mesh.triangles_tree
    vertices, faces = truth['mesh']
    tree = truth['_triangle_tree']
    extent = np.array([radius+padding, radius+padding, vertical+padding])
    for point in path:
        ids = list(tree.intersection(np.r_[point-extent, point+extent]))
        if not ids:
            continue
        section = horizontal_section(vertices, faces[ids], point[2]-extent[2], point[2]+extent[2])
        if projected_clearance([point[:2]], section) <= extent[0]:
            return True
    return False


def mesh_clearance(path, truth, vertical, horizon=2., spacing=.1):
    """Smallest horizontal distance from the path to truth surfaces in its body height band.

    Offline only; capped at ``horizon``. Used to score a flown trajectory against
    a body envelope other than the simulated one (e.g. the deployment airframe).
    The path is resampled by arc length: a 100 Hz trace is far denser than needed.
    """
    path = np.asarray(path, dtype=float).reshape(-1, 3)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if len(path) > 1 and arc[-1] > 0:
        keep = np.r_[True, np.diff(arc) > 0]
        stations = np.linspace(0., arc[-1], max(2, int(np.ceil(arc[-1]/spacing))+1))
        path = np.stack([np.interp(stations, arc[keep], path[keep, i]) for i in range(3)], axis=1)
    if '_triangle_tree' not in truth:
        import trimesh
        vertices, faces = truth['mesh']
        truth['_triangle_tree'] = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).triangles_tree
    vertices, faces = truth['mesh']
    tree, extent, best = truth['_triangle_tree'], np.array([horizon, horizon, vertical]), float(horizon)
    for point in path:
        ids = list(tree.intersection(np.r_[point-extent, point+extent]))
        if ids:
            section = horizontal_section(vertices, faces[ids], point[2]-vertical, point[2]+vertical)
            best = min(best, projected_clearance([point[:2]], section))
    return best


def truth_collision(path, truth, radius, vertical):
    if 'mesh' in truth:
        return mesh_collision(path, truth, radius, vertical)
    trees = np.asarray(truth.get('trees', []), dtype=float).reshape(-1, 4)
    for point in path:
        relevant = (point[2]+vertical+.025 >= 0.) & (point[2]-vertical-.025 <= trees[:, 3])
        if (np.linalg.norm(trees[relevant, :2]-point[:2], axis=1)
                <= trees[relevant, 2]+radius+.025).any():
            return True
    return False


def observed_clearance(path, points):
    distance = float('inf')
    for point in path:
        nearby = points[np.abs(points[:, 2]-point[2]) <= VERTICAL_RADIUS+.025]
        if len(nearby):
            distance = min(distance, float(np.linalg.norm(nearby[:, :2]-point[:2], axis=1).min()))
    return distance


def reference_clearance_audit(path, truth, clearance):
    """Independent mesh-surface metric, including half the sampling spacing."""
    if 'mesh' not in truth:
        raise ValueError('reference safety profile requires a truth mesh for offline audit')
    if '_surface_mesh' not in truth:
        import trimesh
        vertices, faces = truth['mesh']
        truth['_surface_mesh'] = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    dense = sample_polyline(path)
    minimum = min(float(truth['_surface_mesh'].nearest.on_surface(chunk)[1].min())
                  for chunk in np.array_split(dense, max(1, int(np.ceil(len(dense)/128)))))
    padding = .5*np.linalg.norm(np.diff(dense, axis=0), axis=1).max(initial=0.)
    result = dict(minimum_m=minimum, required_m=float(clearance+padding),
                  passed=bool(minimum > clearance+padding))
    if 'terrain' in truth:
        from forest_generator.terrain import grid_height
        terrain = truth['terrain']
        agl = dense[:, 2]-grid_height(dense, terrain['xs'], terrain['ys'], terrain['heights'])
        low, high = truth['agl_limits']
        height_ok = bool((agl > low).all() and (agl < high).all())
        result.update(minimum_agl_m=float(agl.min()), maximum_agl_m=float(agl.max()), height_limits_pass=height_ok)
        result['passed'] &= height_ok
    return result


def evaluate_path(raw, observation, truth):
    protocol = int(observation.get('protocol_version', PROTOCOL_VERSION))
    if protocol != PROTOCOL_VERSION:
        return dict(status='INVALID', protocol_valid=False, error='evaluation requires XYZ protocol 3')
    try:
        parsed = parse_path(raw, protocol_version=protocol)
    except ValueError as error:
        return dict(status='INVALID', protocol_valid=False, error=str(error))
    result = dict(status=parsed['status'], protocol_valid=True)
    points = np.asarray(observation['points'])
    corridor = observation.get('corridor', [])
    flight = (SimpleNamespace(**json.loads(str(observation['flight_json'])))
              if 'flight_json' in observation else DEFAULT_FLIGHT)
    evidence = observation.get('evidence', np.empty((0, 3)))
    safety = PathSafety(points, evidence, corridor if len(corridor) else None,
                        flight, float(observation.get('lidar_hz', 10.)))
    reference_clearance = center_clearance(flight)
    if parsed['status'] in ('HOLD', 'STOP'):
        if reference_clearance is not None:
            brake = QuinticPath.brake(np.zeros(3), observation['velocity'], observation['acceleration'],
                                     flight.max_acceleration, flight.max_jerk)
            stop = brake.remaining(0., np.zeros(3))
            checks = safety.checks(stop, np.zeros(3))
            world = local_to_world(stop, observation['position'], float(observation['yaw']))
            audit = reference_clearance_audit(world, truth, reference_clearance)
            result.update(stop_checks=checks, safety_filter_accept=all(checks.values()),
                          reference_clearance=audit, truth_margin_violation=not audit['passed'],
                          truth_collision=bool(truth_collision(sample_polyline(world), truth, RADIUS, BODY_VERTICAL_RADIUS)))
        return result
    predicted = np.asarray(parsed['points'])
    path = np.vstack([np.zeros(3), predicted])
    # 原始六点与控制器重建曲线都独立检查真值；重建只读取观测，不读取网格。
    curve = None
    try:
        curve = safety.reconstruct(path, observation.get('velocity'), observation.get('acceleration'), np.zeros(3))
    except ValueError as error:
        result['execution_error'] = str(error)
    dense = sample_polyline(curve.remaining(0., np.zeros(3)) if curve is not None else path)
    corridor_ok = not len(corridor) or bool(inside_polygon(dense[:, :2], corridor).all())
    observed = observed_clearance(dense, points)
    world = local_to_world(dense, observation['position'], float(observation['yaw']))
    raw_world = local_to_world(sample_polyline(path), observation['position'], float(observation['yaw']))
    collision = truth_collision(world, truth, RADIUS, BODY_VERTICAL_RADIUS)
    margin = truth_collision(world, truth, INFLATION, VERTICAL_RADIUS) if reference_clearance is None else False
    if reference_clearance is not None:
        audit = reference_clearance_audit(world, truth, reference_clearance)
        result['reference_clearance'] = audit
        margin = not audit['passed']
        if curve is not None:
            braking = safety.braking_corridor(curve, 0., np.zeros(3))
            brake_world = local_to_world(braking, observation['position'], float(observation['yaw']))
            result['braking_reference_clearance'] = reference_clearance_audit(brake_world, truth, reference_clearance)
    goal = np.asarray(observation['goal'])
    known = safety.known(dense, np.zeros(3))
    accepted = curve is not None and corridor_ok and known and safety.guard(curve, 0., np.zeros(3))
    result.update(observed_clearance=observed if np.isfinite(observed) else None,
        observed_body_clearance=observed-RADIUS if np.isfinite(observed) else None,
        truth_collision=bool(collision), truth_margin_violation=bool(margin),
        raw_truth_collision=bool(truth_collision(raw_world, truth, RADIUS, BODY_VERTICAL_RADIUS)),
        truth_geometry='body_cylinder_and_reference_sphere_v1' if reference_clearance is not None else 'swept_cylinder_xyz_v1', execution_reconstructed=curve is not None,
        corridor_violation=not corridor_ok, unknown_path=not known, safety_filter_accept=bool(accepted),
        path_length=float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()),
        endpoint_goal_error=float(np.linalg.norm(path[-1]-goal)),
        endpoint_height_error=float(abs(path[-1, 2]-goal[2])))
    return result

def summarize(results):
    n = len(results)
    if not n: raise ValueError('empty evaluation split')
    go = [r for r in results if r['status']=='GO']
    summary = dict(samples=n, protocol_valid_rate=sum(r['protocol_valid'] for r in results)/n,
                   go_coverage=len(go)/n)
    for s in ['GO','HOLD','STOP','INVALID']:
        summary[s.lower()+'_ratio'] = sum(r['status']==s for r in results)/n
    for metric in ['truth_collision','raw_truth_collision','truth_margin_violation','corridor_violation','unknown_path',
                   'observed_clearance','observed_body_clearance','path_length','endpoint_goal_error',
                   'endpoint_height_error']:
        values = [r[metric] for r in go if r.get(metric) is not None]
        summary[metric+'_go_mean'] = float(np.mean(values)) if values else None
    accepted = [r for r in go if r['safety_filter_accept']]
    summary['filtered_go_coverage'] = len(accepted)/n
    summary['filtered_truth_collision_rate'] = float(np.mean([r['truth_collision'] for r in accepted])) if accepted else None
    times = [r['latency_seconds'] for r in results]
    summary.update(latency_mean=float(np.mean(times)), latency_p95=float(np.percentile(times,95)))
    labelled = [r for r in results if 'target_status' in r]
    if labelled:
        predicted_stop = [r for r in labelled if r['status']=='STOP']
        target_stop = [r for r in labelled if r['target_status']=='STOP']
        correct_stop = sum(r['target_status']=='STOP' for r in predicted_stop)
        summary.update(status_accuracy=float(np.mean([r['status']==r['target_status'] for r in labelled])),
            stop_precision=correct_stop/len(predicted_stop) if predicted_stop else None,
            stop_recall=correct_stop/len(target_stop) if target_stop else None)
        target_go = [r for r in labelled if r['target_status']=='GO']
        waiting = [r for r in labelled if r['status'] in ('HOLD', 'STOP')]
        target_waiting = [r for r in labelled if r['target_status'] in ('HOLD', 'STOP')]
        summary.update(false_hold_rate=(sum(r['status'] in ('HOLD', 'STOP') for r in target_go)/len(target_go)
                                        if target_go else None),
            hold_precision=(sum(r['target_status'] in ('HOLD', 'STOP') for r in waiting)/len(waiting)
                            if waiting else None),
            hold_recall=(sum(r['status'] in ('HOLD', 'STOP') for r in target_waiting)/len(target_waiting)
                         if target_waiting else None))
    totals = [r['total_seconds'] for r in results if 'total_seconds' in r]
    if totals:
        summary['total_seconds_mean'] = float(np.mean(totals))
    return summary


def summarize_scenes(results):
    """每个场景等权；无有效分母的指标为 None，并报告有效场景数。"""
    groups = {}
    for result in results:
        groups.setdefault(result['scene_id'], []).append(result)
    per_scene = {key: summarize(rows) for key, rows in groups.items()}
    macro, counts = {}, {}
    for key in next(iter(per_scene.values()), {}):
        if key == 'samples':
            continue
        values = [s[key] for s in per_scene.values() if s.get(key) is not None]
        macro[key] = float(np.mean(values)) if values else None
        counts[key] = len(values)
    return dict(count=len(groups), macro=macro, valid_scene_counts=counts, per_scene=per_scene)
