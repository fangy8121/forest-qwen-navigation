"""从局部三维观测生成六点 XYZ 教师轨迹，与执行端共用碰撞和制动检查。"""
import numpy as np
from scipy.ndimage import label
from types import SimpleNamespace
from .esdf import astar
from .geometry import (BOUNDS, Z_BOUNDS, GRID_RESOLUTION, INFLATION, VERTICAL_RADIUS,
                       inside, inside_polygon, occupied_centers, center_clearance)
from .planning import PathSafety, smooth_path
from .trajectory import QuinticPath

DEFAULT_FLIGHT = SimpleNamespace(max_speed=1., max_acceleration=1., max_jerk=4.,
    max_yaw_rate=1., max_yaw_acceleration=2., corner_distance=1.5, max_path_deviation=.3)


def resample_six(path):
    p = np.asarray(path)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    ts = np.linspace(0, lengths[-1], 7)[1:]
    return np.column_stack([np.interp(ts, lengths, p[:, d]) for d in range(p.shape[1])])


def teacher_path(points, goal, flight=DEFAULT_FLIGHT, start_vel=None, start_acc=None,
                 corridor=None, evidence=None, lidar_hz=10.):
    hold = {'status': 'HOLD', 'points': []}
    if not len(points) and (evidence is None or not len(evidence)):
        return hold, 'empty_observation'
    if evidence is None or not len(evidence):
        return hold, 'insufficient_observed_braking_space'
    goal = np.asarray(goal, dtype=float)
    if (goal.shape != (3,) or not np.isfinite(goal).all() or not inside([goal[:2]])
            or not Z_BOUNDS[0] <= goal[2] <= Z_BOUNDS[1] or np.linalg.norm(goal) < .3):
        return hold, 'goal_unavailable'
    safety = PathSafety(points, evidence, corridor, flight, lidar_hz)
    velocity = np.zeros(3) if start_vel is None else np.asarray(start_vel)
    acceleration = np.zeros(3) if start_acc is None else np.asarray(start_acc)
    brake = QuinticPath.brake(np.zeros(3), velocity, acceleration,
                             flight.max_acceleration, flight.max_jerk)
    braking = brake.remaining(0., np.zeros(3))
    if not safety.clear(braking):
        return hold, 'blocked_braking_space'
    if not safety.known(braking, np.zeros(3)):
        return hold, 'insufficient_observed_braking_space'

    def safe(path):
        return safety.clear(path) and safety.known(path, np.zeros(3))

    step = GRID_RESOLUTION
    lower = np.array([BOUNDS[0], BOUNDS[2], np.ceil(Z_BOUNDS[0]/step)*step])
    upper = np.array([BOUNDS[1], BOUNDS[3], np.floor(Z_BOUNDS[1]/step)*step])
    shape = np.rint((upper-lower)/step).astype(int)+1
    cells = np.indices(shape).reshape(3, -1).T
    centers = lower+cells*step
    free = ~occupied_centers(safety.obstacles, centers, INFLATION+step/np.sqrt(2),
                             VERTICAL_RADIUS+step/2)
    if corridor is not None:
        free &= inside_polygon(centers[:, :2], corridor)
    free[free] &= safety.known_points(centers[free], np.zeros(3))
    free = free.reshape(shape)
    start = np.rint(-lower/step).astype(np.int64)
    end = np.clip(np.rint((goal-lower)/step).astype(np.int64), 0, shape-1)
    if not free[tuple(start)] or not safe(np.zeros((1, 3))):
        return hold, 'start_blocked'
    # 六邻域连通分量与禁止穿角的 A* 使用相同可达集合，候选目标只搜索一次。
    connected, _ = label(free)
    reachable = connected == connected[tuple(start)]
    penalty = (.03*np.abs(centers[:, 2]-goal[2])).reshape(shape).astype(np.float32)
    reason = 'no_reachable_subgoal'
    if reachable[tuple(end)]:
        route = lower+np.asarray(astar(free, penalty, start, end))*step
        answer, reason = route_answer(np.vstack([route, goal]), safe, safety,
                                      flight, start_vel, start_acc)
        if answer is not None:
            return answer, reason
    projection = centers @ (goal/np.linalg.norm(goal))
    candidates = np.flatnonzero(reachable.ravel() & (projection >= .5)
                               & (np.linalg.norm(centers, axis=1) >= 1.))
    score = (np.linalg.norm(centers[candidates]-goal, axis=1)
             + .1*np.linalg.norm(centers[candidates], axis=1))
    tried = [goal]
    for index in candidates[np.argsort(score)]:
        target = centers[index]
        if min(np.linalg.norm(target-p) for p in tried) < .8:
            continue
        tried.append(target)
        route = lower+np.asarray(astar(free, penalty, start, cells[index]))*step
        answer, reason = route_answer(route, safe, safety, flight, start_vel, start_acc)
        if answer is not None:
            return answer, 'reachable_subgoal'
        if len(tried) >= 9:
            break
    return hold, reason


def route_answer(route, safe, safety, flight, start_vel, start_acc, shortcut_check=None):
    p = np.vstack([np.zeros(3), route])
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]]
    simple, i = [p[0]], 0
    def shortcut(a, b):
        samples = np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/.08))+1))
        clear = (safety.clear(samples) if center_clearance(flight) is not None
                 else safety.clear_points(samples, padding=GRID_RESOLUTION/2).all())
        return (clear
                and safety.known(samples, np.zeros(3))
                and (shortcut_check is None or shortcut_check(samples)))
    while i < len(p)-1:
        j = len(p)-1
        while j > i+1 and not shortcut(p[i], p[j]):
            j -= 1
        simple.append(p[j]); i = j
    try:
        curve = smooth_path(np.asarray(simple), flight, safe, start_vel, start_acc)
    except ValueError:
        return None, 'smoothing_failure'
    dense = curve.remaining(0., np.zeros(3))
    reason = 'reconstruction_failure'
    for candidates in six_point_candidates(dense):
        six = np.round(candidates, 4)
        reconstructed = np.vstack([np.zeros(3), six])
        if not safe(reconstructed):
            continue
        try:
            rebuilt = smooth_path(reconstructed, flight, safe, start_vel, start_acc)
        except ValueError:
            continue
        if safety.guard(rebuilt, 0., np.zeros(3)):
            return {'status': 'GO', 'points': six.tolist()}, 'executable_path'
        reason = 'braking_space_unavailable'
    return None, reason


def six_point_candidates(dense):
    """Uniform arc length plus curvature-aware compression of the same curve."""
    indices = [0, len(dense)-1]
    for _ in range(5):
        choices = []
        for left, right in zip(indices[:-1], indices[1:]):
            if right-left < 2:
                continue
            delta = dense[right]-dense[left]
            t = np.clip((dense[left:right+1]-dense[left]) @ delta / max(delta @ delta, 1e-12), 0, 1)
            error = np.linalg.norm(dense[left:right+1]-dense[left]-t[:, None]*delta, axis=1)
            middle = left+int(np.argmax(error)) if error.max() > .001 else (left+right)//2
            choices.append((float(error.max())+.0001*(right-left), middle))
        if not choices:
            break
        indices.append(max(choices)[1]); indices.sort()
    alternatives = [resample_six(dense)]
    if len(indices) == 7:
        alternatives.append(dense[indices[1:]])
    return alternatives
