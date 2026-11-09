"""Visible BEV labels and safe horizontal counterfactual observations."""
import json
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (BOUNDS, HEIGHT_LEVELS, GRID_RESOLUTION, INFLATION,
                       VERTICAL_RADIUS, inside_polygon, observed_body_height,
                       observed_centers, parse_path, local_to_world, world_to_local)
from .supervision import MANEUVER_THRESHOLD, parse_prompt_goal


SPATIAL_STATES = ('blocked', 'observed', 'unknown')
GROUNDING_VERSION = 3


def task_kind(row):
    task = row.get('task', 'path')
    if task not in ('path', 'spatial'):
        raise ValueError(f'unknown training task: {task}')
    return task


def parse_spatial(text):
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError('duplicate spatial JSON field')
        return dict(pairs)
    try:
        result = json.loads(text, object_pairs_hook=unique)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError('invalid spatial JSON') from error
    if (not isinstance(result, dict) or set(result) != {'state'}
            or result['state'] not in SPATIAL_STATES):
        raise ValueError('spatial answer requires exactly state: blocked/observed/unknown')
    return result


def maneuver_group(row):
    """Descriptive teacher deviation, including a straight offset subgoal.

    These groups do not claim that one direction is the only safe action.
    The goal ray, rather than the route's endpoint chord, defines straight.
    """
    answer = parse_path(row['answer'])
    if answer['status'] == 'HOLD':
        return 'hold'
    goal = np.asarray(parse_prompt_goal(row['prompt']), dtype=float)
    points = np.asarray(answer['points'], dtype=float)
    if goal @ goal < 1e-12:
        raise ValueError('GO sampling requires a nonzero prompt goal')
    residual = points - np.outer(points @ goal / (goal @ goal), goal)
    y, z = residual[:, 1], residual[:, 2]
    lateral, vertical = np.max(np.abs(y)), np.max(np.abs(z))
    if max(lateral, vertical) < MANEUVER_THRESHOLD:
        return 'along_goal'
    if lateral >= vertical:
        return 'left' if y[np.argmax(np.abs(y))] > 0 else 'right'
    return 'up' if z[np.argmax(np.abs(z))] > 0 else 'down'


def spatial_prompt(center):
    x, y, z = np.round(center, 3).tolist()
    return ('读取无人机局部多高度BEV：x前/y左/z上，单位米；各图共用XY坐标。'
            f'从左到右五张主图的中心高度为{list(HEIGHT_LEVELS)}m。'
            '橙色为考虑机体和余量后的障碍区，浅色为已观测空间，灰色为未知空间。'
            '右侧Lower hits小图只显示低处障碍命中，灰底不能证明可通行。'
            f'只读取中心高度z={z}m的主图，位置XY=[{x},{y}]的栅格属于哪类？'
            'blocked表示橙色障碍，observed表示浅色已观测，unknown表示灰色未知。'
            '不要推断整条路线是否安全，不要输出解释或代码围栏。'
            '只输出一个JSON对象：{"state":"blocked"}、{"state":"observed"}或{"state":"unknown"}。')


def visible_states(points, evidence, centers, max_acceleration=1.):
    """Exactly the colors of main BEV cells; no hidden mesh or actual-body exception."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    # Snap only XY: shown heights include .5 m, while the ray grid is .2 m.
    snapped = np.round(centers[:, :2]/GRID_RESOLUTION)*GRID_RESOLUTION
    if (not np.allclose(centers[:, :2], snapped, atol=1e-8, rtol=0)
            or not np.isin(centers[:, 2], HEIGHT_LEVELS).all()
            or not inside_polygon(centers[:, :2], bounds_polygon()).all()):
        raise ValueError('spatial queries must be main BEV grid centers inside XY bounds')
    seen = observed_centers(evidence, centers, vertical=observed_body_height(max_acceleration))
    states = np.where(seen, 'observed', 'unknown')
    for height in np.unique(centers[:, 2]):
        mask = centers[:, 2] == height
        relevant = points[np.abs(points[:, 2]-height) <= VERTICAL_RADIUS]
        if len(relevant):
            blocked = cKDTree(relevant[:, :2]).query(centers[mask, :2])[0] <= INFLATION
            indices = np.flatnonzero(mask)
            states[indices[blocked]] = 'blocked'
    return states


def spatial_query_states(observation, candidates):
    """Return visible center colors and a mask of unambiguous, unobscured queries."""
    candidates = np.asarray(candidates, dtype=float).reshape(-1, 3)
    # Avoid boundaries and one-cell speckles: all nine nearby centers agree.
    offsets = np.array([[dx, dy, 0.] for dx in (-.2, 0., .2) for dy in (-.2, 0., .2)])
    neighbors = (candidates[:, None, :]+offsets).reshape(-1, 3)
    flight = json.loads(str(observation['flight_json']))
    states = visible_states(observation['points'], observation['evidence'], neighbors,
                            flight['max_acceleration']).reshape(len(candidates), -1)
    uniform = (states == states[:, :1]).all(axis=1)
    # Goal/star, origin/arrow and corridor overlays must not obscure a query.
    uniform &= np.linalg.norm(candidates[:, :2], axis=1) > 1.
    uniform &= np.linalg.norm(candidates[:, :2]-observation['goal'][:2], axis=1) > .7
    corridor = observation.get('corridor')
    if corridor is not None and len(corridor):
        from .geometry import segment_distances
        boundary_distance = np.min([segment_distances(candidates[:, :2], a, b)
                                    for a,b in zip(corridor, np.roll(corridor, -1, axis=0))], axis=0)
        uniform &= boundary_distance > .3
    return states[:, 0], uniform


def spatial_examples(observation, seed):
    """At most one unambiguous query per state, chosen across all shown heights."""
    candidates = np.array([[x, y, z] for z in HEIGHT_LEVELS
                           for x in (2., 4., 6.) for y in (-2., 0., 2.)])
    states, uniform = spatial_query_states(observation, candidates)
    rng = np.random.default_rng(seed)
    examples = []
    for state in SPATIAL_STATES:
        indices = np.flatnonzero(uniform & (states == state))
        if len(indices):
            center = candidates[rng.choice(indices)]
            examples.append(dict(query=center.tolist(), prompt=spatial_prompt(center),
                                 answer=json.dumps({'state': state}, separators=(',', ':'))))
    return examples


def bounds_polygon():
    x0, x1, y0, y1 = BOUNDS
    return np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]], dtype=float)


def _ccw(polygon):
    polygon = np.asarray(polygon, dtype=float)
    area = np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                  - polygon[:, 1]*np.roll(polygon[:, 0], -1))
    return polygon if area >= 0 else polygon[::-1]


def polygon_intersection(subject, clip):
    """Convex Sutherland-Hodgman clipping, accepting either winding."""
    result = _ccw(subject)
    for a,b in zip(_ccw(clip), np.roll(_ccw(clip), -1, axis=0)):
        if not len(result):
            break
        edge = b-a
        def side(p):
            delta = p-a
            return edge[0]*delta[1]-edge[1]*delta[0]
        output = []
        previous = result[-1]
        previous_side = side(previous)
        for current in result:
            current_side = side(current)
            if (current_side >= 0) != (previous_side >= 0):
                output.append(previous+(current-previous)*previous_side/(previous_side-current_side))
            if current_side >= 0:
                output.append(current)
            previous, previous_side = current, current_side
        result = np.asarray(output, dtype=float).reshape(-1, 2)
    # Remove only adjacent duplicates from intersections touching corners.
    if len(result):
        result = result[np.linalg.norm(result-np.roll(result, 1, axis=0), axis=1) > 1e-9]
    return result


def reflected_mesh(vertices, faces, observation, reflection):
    local = world_to_local(vertices, observation['position'], float(observation['yaw']))
    vertices = local_to_world(local@reflection.T, observation['position'], float(observation['yaw']))
    # A reflection reverses orientation; preserve consistent surface normals.
    return vertices, np.asarray(faces)[:, ::-1]


def route_observation_checks(observation, answer):
    """Check a stationary route using only observed geometry and ray evidence."""
    from .evaluation import sample_polyline
    from .planning import PathSafety, smooth_path
    flight = SimpleNamespace(**json.loads(str(observation['flight_json'])))
    parsed = parse_path(answer)
    if parsed['status'] != 'GO':
        raise ValueError('paired checks require a GO route')
    safety = PathSafety(observation['points'], observation['evidence'], observation['corridor'],
                        flight, float(observation.get('lidar_hz', 10.)))
    path = np.vstack([np.zeros(3), parsed['points']])
    clear = safety.clear(path)
    known = safety.known(sample_polyline(path), np.zeros(3))
    accepted = False
    try:
        curve = smooth_path(path, flight, lambda p: safety.clear(p) and safety.known(p, np.zeros(3)),
                            np.zeros(3), np.zeros(3))
        dense = sample_polyline(curve.remaining(0., np.zeros(3)))
        accepted = safety.clear(dense) and safety.known(dense, np.zeros(3)) and safety.guard(curve, 0., np.zeros(3))
    except ValueError:
        pass
    return dict(observed_clear=bool(clear), observed_known=bool(known), accepted=bool(accepted))


def pair_contrast(pair):
    """Both candidate routes must require their own image, not be interchangeable."""
    if len(pair) != 2:
        raise ValueError('a counterfactual pair requires exactly two observations')
    if any(min(point[2] for point in parse_path(item['answer'])['points']) < min(HEIGHT_LEVELS)
           for item in pair):
        return None, 'paired_route_below_main_display'
    cross = [route_observation_checks(item['observation'], pair[1-index]['answer'])
             for index,item in enumerate(pair)]
    if any(item['accepted'] for item in cross):
        return None, 'paired_routes_not_image_dependent'
    # Swapped teacher choices alone are insufficient: a trivial common goal
    # interpolation can still defeat the intended straight-path diagnosis.
    goal = np.asarray(pair[0]['observation']['goal'], dtype=float)
    straight = json.dumps(dict(status='GO', points=np.outer(np.arange(1,7)/6,goal).tolist()),
                          separators=(',', ':'))
    interpolated = [route_observation_checks(item['observation'], straight) for item in pair]
    if all(item['accepted'] for item in interpolated):
        return None, 'paired_common_goal_interpolation'
    for negative,check in zip(cross,interpolated):
        negative['goal_interpolation'] = check
    return cross, 'validated_image_contrast'


def counterfactual_pair(observation):
    """Replan stationary derivatives inside BOTH original/reflected task corridors.

    Both records use the same origin, goal, corridor, zero velocity/acceleration.
    Reflection is horizontal; height, gravity and lower-hit meaning are retained.
    Validation also has to check the independently transformed truth mesh.
    """
    from .planning import PathSafety, smooth_path
    from .qwen import task_prompt
    from .teacher import teacher_path
    goal = np.asarray(observation['goal'], dtype=float)
    norm = np.linalg.norm(goal[:2])
    if norm < .3:
        return None, 'goal_has_no_horizontal_direction'
    direction = goal[:2]/norm
    reflection = np.eye(3)
    reflection[:2, :2] = 2*np.outer(direction, direction)-np.eye(2)
    corridor = np.asarray(observation.get('corridor', bounds_polygon()))
    if not len(corridor):
        corridor = bounds_polygon()
    common = polygon_intersection(corridor, corridor@reflection[:2, :2].T)
    common = polygon_intersection(common, bounds_polygon())
    common = polygon_intersection(common, bounds_polygon()@reflection[:2, :2].T)
    if len(common) < 3 or not inside_polygon(np.array([[0.,0.], goal[:2]]), common).all():
        return None, 'no_common_corridor'
    flight = SimpleNamespace(**json.loads(str(observation['flight_json'])))
    hz = float(observation.get('lidar_hz', 10.))
    answer, reason = teacher_path(observation['points'], goal, flight, np.zeros(3), np.zeros(3),
                                 common, observation['evidence'], hz)
    if answer['status'] != 'GO':
        return None, reason
    teacher = np.asarray(answer['points'])
    separation = float(np.linalg.norm(teacher-teacher@reflection.T, axis=1).max())
    if separation < .4:
        return None, 'route_pair_not_distinct'
    prompt = task_prompt(goal, corridor=common, velocity=np.zeros(3), acceleration=np.zeros(3))
    pair = []
    for name, transform in (('stationary', np.eye(3)), ('reflected', reflection)):
        derived = {key: np.copy(value) for key,value in observation.items()}
        derived.update(points=observation['points']@transform.T,
                       evidence=observation['evidence']@transform.T, corridor=common,
                       velocity=np.zeros(3), acceleration=np.zeros(3))
        route = np.round(teacher@transform.T, 4)
        text = json.dumps(dict(status='GO', points=route.tolist()), separators=(',', ':'))
        parse_path(text)
        safety = PathSafety(derived['points'], derived['evidence'], common, flight, hz)
        path = np.vstack([np.zeros(3), route])
        try:
            curve = smooth_path(path, flight, lambda p: safety.clear(p) and safety.known(p, np.zeros(3)),
                                np.zeros(3), np.zeros(3))
            valid = safety.clear(path) and safety.known(path, np.zeros(3)) and safety.guard(curve, 0., np.zeros(3))
        except ValueError:
            valid = False
        if not valid:
            return None, 'paired_route_failed_observation_or_dynamics'
        pair.append(dict(variant=name, observation=derived, transform=transform, prompt=prompt,
                         answer=text, teacher_reason=reason, separation=separation))
    cross, contrast_reason = pair_contrast(pair)
    if cross is None:
        return None, contrast_reason
    for item, negative in zip(pair, cross):
        item['cross_route_checks'] = negative
    return pair, 'validated_observation_pair'
