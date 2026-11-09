import json
import numpy as np
import pytest
from omni_drones.navigation.forest.teacher import teacher_path, DEFAULT_FLIGHT
from omni_drones.navigation.forest.planning import PathSafety, smooth_path
from omni_drones.navigation.forest.geometry import local_to_world, parse_path
from omni_drones.navigation.forest.esdf import astar
from omni_drones.navigation.forest.qwen import task_prompt
from omni_drones.navigation.forest.observation import SampleSelector
from omni_drones.navigation.forest.evaluation import evaluate_path, mesh_collision, summarize


def evidence_grid():
    axes = np.meshgrid(np.arange(-2, 8.1, .2), np.arange(-2, 2.1, .2),
                       np.arange(-2, 2.1, .2), indexing='ij')
    return np.column_stack([a.ravel() for a in axes])


def wall(x, upper):
    y, z = np.meshgrid(np.arange(-1.4, 1.41, .15),
                       np.arange(0., 2.01, .15) if upper else np.arange(-2., .01, .15))
    return np.c_[np.full(y.size, x), y.ravel(), z.ravel()]


@pytest.mark.parametrize('upper', [True, False])
def test_same_xy_obstacle_requires_opposite_vertical_route(upper):
    points = wall(3., upper)
    corridor = np.array([[-2., -.8], [8., -.8], [8., .8], [-2., .8]])
    evidence = evidence_grid()
    answer, reason = teacher_path(points, [6., 0., 0.], evidence=evidence, corridor=corridor)
    assert answer['status'] == 'GO', reason
    path = np.vstack([np.zeros(3), answer['points']])
    safety = PathSafety(points, evidence, corridor, DEFAULT_FLIGHT)
    curve = smooth_path(path, DEFAULT_FLIGHT, safety.clear)
    dense = curve.remaining(0., np.zeros(3))
    crossing = dense[np.abs(dense[:, 0]-3.) < .15]
    assert len(crossing)
    assert (crossing[:, 2] < -.4).all() if upper else (crossing[:, 2] > .4).all()
    assert safety.clear(dense) and safety.guard(curve, 0., np.zeros(3))
    assert np.linalg.norm(path[-1]-[6., 0., 0.]) < .1


def test_alternating_upper_lower_obstacles_reconstruct_safely():
    points = np.vstack([wall(2., True), wall(4.5, False)])
    corridor = np.array([[-2., -.8], [8., -.8], [8., .8], [-2., .8]])
    evidence = evidence_grid()
    answer, reason = teacher_path(points, [7., 0., 0.], evidence=evidence, corridor=corridor)
    assert answer['status'] == 'GO', reason
    path = np.vstack([np.zeros(3), answer['points']])
    safety = PathSafety(points, evidence, corridor, DEFAULT_FLIGHT)
    curve = smooth_path(path, DEFAULT_FLIGHT, safety.clear)
    dense = curve.remaining(0., np.zeros(3))
    assert safety.clear(dense)
    assert dense[np.abs(dense[:, 0]-2.) < .15, 2].max() < -.4
    assert dense[np.abs(dense[:, 0]-4.5) < .15, 2].min() > .4


def test_unknown_and_full_height_blockage_do_not_generate_a_crossing():
    points = np.vstack([wall(2., True), wall(2., False)])
    corridor = np.array([[-2., -.8], [8., -.8], [8., .8], [-2., .8]])
    answer, _ = teacher_path(points, [6., 0., 0.], evidence=np.empty((0, 3)))
    assert answer == {'status': 'HOLD', 'points': []}
    answer, _ = teacher_path(points, [6., 0., 0.], evidence=evidence_grid(), corridor=corridor)
    assert answer['status'] == 'HOLD' or max(p[0] for p in answer['points']) < 1.3


def test_true_mesh_height_and_frame_are_checked_independently():
    local_vertices = np.array([[3., -.3, .8], [3., .3, .8], [3., .3, 1.2], [3., -.3, 1.2]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    position, yaw = np.array([7., 2., 4.]), .8
    truth = {'mesh': (local_to_world(local_vertices, position, yaw), faces)}
    observation = dict(points=np.empty((0, 3)), evidence=evidence_grid(), position=position,
                       yaw=yaw, goal=np.array([6., 0., 0.]), velocity=np.zeros(3), acceleration=np.zeros(3))
    low = json.dumps(dict(status='GO', points=[[x, 0., 0.] for x in range(1, 7)]))
    high = json.dumps(dict(status='GO', points=[[x, 0., 1.] for x in range(1, 7)]))
    low_result = evaluate_path(low, observation, truth)
    high_result = evaluate_path(high, observation, truth)
    assert not low_result['truth_collision'] and low_result['safety_filter_accept']
    assert high_result['truth_collision'] and high_result['raw_truth_collision']
    assert high_result['safety_filter_accept']  # 真值隐藏障碍不参与观测安全筛选。
    assert high_result['endpoint_height_error'] == 1.


def test_thin_obstacle_between_clear_height_layers_blocks_transition():
    vertices = np.array([[.35, -.2, .5], [.65, -.2, .5], [.5, .2, .5]])
    truth = {'mesh': (vertices, np.array([[0, 1, 2]]))}
    assert not mesh_collision(np.array([[0., 0., 0.], [1., 0., 1.]]), truth, .5, .25)
    observation = dict(points=np.empty((0, 3)), evidence=evidence_grid(), position=np.zeros(3),
                       yaw=0., goal=np.array([6., 0., 1.]))
    raw = json.dumps(dict(status='GO', points=[[x, 0., 1.] for x in range(1, 7)]))
    result = evaluate_path(raw, observation, truth)
    assert result['raw_truth_collision'] and result['truth_collision']


def test_evaluation_rejects_old_height_clipped_observation():
    result = evaluate_path('{"status":"HOLD","points":[]}', {'protocol_version': 2}, {})
    assert result['status'] == 'INVALID' and not result['protocol_valid']


def evidence_full_grid():
    axes = np.meshgrid(np.arange(-2,10.1,.2), np.arange(-6,6.1,.2), np.arange(-2,2.1,.2), indexing='ij')
    return np.column_stack([a.ravel() for a in axes])


def test_blocked_goal_selects_executable_subgoal():
    points = np.array([[6.,0.,0.]])
    evidence = evidence_full_grid()
    velocity = np.array([.4,0.,0.])
    answer, reason = teacher_path(points,[6.,0.,0.],start_vel=velocity,evidence=evidence)
    assert answer['status'] == 'GO' and reason == 'reachable_subgoal'
    path = np.vstack([[0.,0.,0.],answer['points']])
    safety = PathSafety(points,evidence,None,DEFAULT_FLIGHT)
    curve = smooth_path(path,DEFAULT_FLIGHT,safety.clear,velocity,np.zeros(3))
    assert safety.clear(curve.remaining(0.,np.zeros(3)))
    assert safety.guard(curve,0.,np.zeros(3))
    assert np.linalg.norm(path[-1]-[6,0,0]) > .5


def test_known_empty_space_is_flyable_but_unknown_space_is_not():
    points = np.empty((0,3))
    answer,_ = teacher_path(points,[6.,0.,0.],evidence=evidence_full_grid())
    assert answer['status'] == 'GO'
    answer,_ = teacher_path(np.array([[6.,4.,0.]]),[6.,0.,0.],evidence=np.empty((0,3)))
    assert answer == {'status':'HOLD','points':[]}


def test_reachable_search_does_not_cut_blocked_corners():
    free = np.zeros((2, 2, 2), dtype=bool)
    free[0,0,0] = free[1,1,1] = True
    with pytest.raises(ValueError, match='no connected'):
        astar(free, np.zeros(free.shape), np.zeros(3, dtype=int), np.ones(3, dtype=int))


def test_xyz_prompt_motion_and_legacy_protocol_rejection():
    prompt = task_prompt(np.array([6.,0.,0.]),velocity=np.array([.4,0,.2]),acceleration=np.zeros(3))
    assert 'HOLD' in prompt and 'STOP' not in prompt and '[0.4, 0.0, 0.2]m/s' in prompt
    with pytest.raises(ValueError, match='protocol 3'):
        task_prompt(np.array([6.,0.]),protocol_version=2)
    assert parse_path('{"status":"HOLD","points":[]}')['status'] == 'HOLD'
    with pytest.raises(ValueError):
        parse_path('{"status":"HOLD","points":[[1,0]]}')


def test_sampling_retains_same_view_at_different_motion_states():
    selector = SampleSelector()
    args = (np.array([[3.,2,0]]),evidence_full_grid(),[6,0,0],[0,0,2],0)
    first = selector.describe(*args,np.zeros(3),np.zeros(3))
    selector.accept(first)
    moving = selector.describe(*args,np.array([.5,0,0]),np.zeros(3))
    assert selector.reason(moving) is None
    assert selector.reason(first) == 'insufficient_change'


def test_false_hold_metric_counts_legacy_and_new_outputs():
    rows = [dict(status=s,target_status='GO',protocol_valid=True,latency_seconds=.1)
            for s in ['STOP','HOLD','INVALID']]
    result = summarize(rows)
    assert result['false_hold_rate'] == 2/3
    assert result['hold_precision'] == 0.
