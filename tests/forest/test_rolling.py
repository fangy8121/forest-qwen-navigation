from concurrent.futures import Future
import json
from types import SimpleNamespace
import numpy as np
import pytest
from omegaconf import OmegaConf
from omni_drones.navigation.forest import planning
from omni_drones.navigation.forest.trajectory import QuinticPath, Heading, connect_path
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


def test_local_smoothing_preserves_seed101_route():
    points = np.array([[0., .22176625280527695, 2.], [7.4, 1.6, 2.],
                       [12., -.12652106999258417, 2.]])
    path = QuinticPath(points, corner_distance=.5, max_path_deviation=.15)
    xyz, velocity, _ = path.sample(np.linspace(0., path.duration, 4000))
    distances = []
    for start, end in zip(points[:-1], points[1:]):
        delta = end-start
        alpha = np.clip((xyz-start) @ delta / (delta @ delta), 0., 1.)
        distances.append(np.linalg.norm(xyz-start-alpha[:, None]*delta, axis=1))
    deviation = np.min(distances, axis=0)
    assert deviation.max() < .15
    straight = np.linalg.norm(xyz-points[1], axis=1) > .51
    assert deviation[straight].max() < 1e-10
    corner = np.argmin(np.linalg.norm(path.points-points[1], axis=1))
    assert np.linalg.norm(path.sample(path.times[corner])[1]) > .1
    assert np.linalg.norm(velocity, axis=1).max() > .8


def test_splice_brake_and_sharp_turn():
    old = QuinticPath([[0,0,2],[2,1,2],[4,0,2]])
    p, v, a = old.sample(1.5)
    new = QuinticPath([p,[2,1,2],[4,1,2]], start_vel=v, start_acc=a)
    brake = QuinticPath.brake(p, v, a)
    for path in (new, brake):
        for expected, actual in zip((p,v,a), path.sample(0)):
            np.testing.assert_allclose(actual, expected, atol=1e-10)
        _, end_v, end_a = path.sample(path.duration)
        np.testing.assert_allclose(end_v, 0, atol=1e-10)
        np.testing.assert_allclose(end_a, 0, atol=1e-10)
    # 原地 180° 折返交给制动/重规划，不制造强行穿过顶点的回环。
    with pytest.raises(ValueError):
        QuinticPath([[0,0,2],[2,0,2],[0,0,2]])


def test_heading_limits_and_consumed_points():
    h = Heading(np.pi-.1, max_rate=.7, max_acceleration=1.2)
    last_yaw, last_rate = h.yaw, h.rate
    for _ in range(200):
        h.step(-np.pi+.1, .01)
        assert abs(h.yaw-last_yaw) <= .007+1e-12
        assert abs(h.rate-last_rate) <= .012+1e-12
        last_yaw, last_rate = h.yaw, h.rate
    assert abs(h.yaw-(np.pi+.1)) < .01
    tail = connect_path([[0,0,2],[1,0,2],[2,0,2],[3,0,2]], [1.4,0,2], [.5,0,0], 1.)
    np.testing.assert_allclose(tail, [[1.4,0,2],[2,0,2],[3,0,2]])


@pytest.mark.parametrize('hold_status', ['HOLD'])
def test_rolling_requests_continue_motion_and_splice(monkeypatch, tmp_path, hold_status):
    now = [10.]
    monkeypatch.setattr(planning.time, 'perf_counter', lambda: now[0])
    cfg = OmegaConf.create(dict(history_seconds=3., view='bev',
        planning=dict(hz=1., max_snapshot_age=3.),
        task=dict(lidar=dict(hz=10.), flight=dict(height=2., max_speed=1., max_acceleration=1.,
            max_jerk=4., max_yaw_rate=1., max_yaw_acceleration=2.,
            corner_distance=.5, max_path_deviation=.15))))
    runner = planning.RollingPlanner(cfg, None, tmp_path, [0,0,2])
    submitted = []
    def submit(fn, model, snap, view):
        snap['folder'].mkdir()
        future = Future()
        submitted.append((snap, future))
        return future
    runner.pool.shutdown()
    runner.pool = SimpleNamespace(submit=submit)
    runner.path = QuinticPath([[0,0,2],[3,0,2],[6,0,2]])
    runner.path_start = 2.
    xx, yy, zz = np.meshgrid(np.arange(-2,10.1,.2), np.arange(-6,6.1,.2), np.arange(-2,2.1,.2))
    evidence = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    state = np.r_[runner.reference(3.)[0], [1.,0,0,0], runner.reference(3.)[1], [0.,0,0]]
    def observe(t):
        runner.observe(dict(points=np.array([[0,4,0.]]), evidence=evidence, position=np.array([0,0,2.]),
                            yaw=0., timestamp=t, goal=np.array([6.,0,0])))
    observe(3.)
    first = runner.tick(3., state, True)
    now[0] = 10.5
    second = runner.tick(3.5, state)
    assert second[0][0] > first[0][0]
    now[0] = 11.
    runner.tick(4., state, True)
    assert len(submitted) == 1 and runner.missed == 1
    snap, future = submitted[0]
    future.set_result(dict(raw=json.dumps(dict(status='GO', points=[[i,0.,.1*i] for i in range(1,7)])),
                           latency_seconds=1.1, pipeline_seconds=1.1))
    now[0] = 11.1
    observe(4.1)
    before = runner.reference(4.1)
    state[:3] = before[0]
    after = runner.tick(4.1, state, True)
    assert runner.path.points[-1, 2] == pytest.approx(2.6)
    assert len(runner.accepted) == 1
    for a,b in zip(before, after):
        np.testing.assert_allclose(a,b,atol=1e-10)
    now[0] = 12.
    observe(5.)
    runner.tick(5., state, True)
    assert len(submitted) == 2 and submitted[1][0]['timestamp'] == 5.
    submitted[1][1].set_result(dict(raw=json.dumps(dict(status=hold_status, points=[])), latency_seconds=.1))
    now[0] = 12.1
    runner.tick(5.1, state)
    assert runner.braking and runner.terminal is None
    assert not runner.needs_snapshot(5.2)
    # 已在途的 GO 不能打断刹停；恢复必须使用停稳后的观测。
    brake = runner.path
    late = tmp_path/'late_go'
    late.mkdir()
    runner.snapshot = dict(submitted[1][0], folder=late)
    runner.future = Future()
    runner.future.set_result(dict(raw=json.dumps(dict(status='GO', points=[[i,0.,0.] for i in range(1,7)])),
                                  latency_seconds=.1))
    runner.tick(5.2, state)
    assert runner.path is brake and json.loads((late/'decision.json').read_text())['reason']=='braking'
    end = runner.path_start+runner.path.duration
    _, end_v, _ = runner.tick(end, state)
    assert runner.path is None
    np.testing.assert_allclose(end_v, 0, atol=1e-10)

    now[0] += end-5.1+1.
    observe(end+.1)
    runner.tick(end+.1, state, True)
    assert len(submitted) == 3
    assert submitted[-1][0]['timestamp'] == end+.1
    submitted[-1][1].set_result(dict(raw=json.dumps(dict(status='GO',points=[[i,0.,.1*i] for i in range(1,7)])),
                                       latency_seconds=.1))
    now[0] += .1
    state[:3] = runner.hover
    runner.tick(end+.2, state)
    assert len(runner.accepted) == 2 and not runner.braking and runner.terminal is None
    runner.stop(end+.2, 'goal_reached', terminal=True)
    assert runner.terminal == 'goal_reached'


def test_hold_observes_after_stopped_replanning_only_once(monkeypatch, tmp_path):
    now = [10.]
    monkeypatch.setattr(planning.time, 'perf_counter', lambda: now[0])
    cfg = OmegaConf.create(dict(history_seconds=3., view='bev',
        planning=dict(hz=1., max_snapshot_age=3., hold_descent=.2),
        task=dict(lidar=dict(hz=10.), flight=dict(vars(DEFAULT_FLIGHT),
                  settle_speed=.15, settle_error=.15))))
    runner = planning.RollingPlanner(cfg, None, tmp_path, [0., 0., 2.])
    submitted = []
    def submit(fn, model, snap, view):
        snap['folder'].mkdir()
        future = Future()
        submitted.append(future)
        return future
    runner.pool.shutdown()
    runner.pool = SimpleNamespace(submit=submit)
    xx, yy, zz = np.meshgrid(np.arange(-2,10.1,.2), np.arange(-6,6.1,.2), np.arange(-2,2.1,.2))
    evidence = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    state = np.r_[[0., 0., 2.], [1., 0., 0., 0.], np.zeros(6)]
    def hold(t):
        runner.observe(dict(points=np.empty((0,3)), evidence=evidence, position=state[:3].copy(),
                            yaw=0., timestamp=t, goal=np.array([6.,0.,0.])))
        runner.tick(t, state, True)
        submitted[-1].set_result(dict(raw='{"status":"HOLD","points":[]}', latency_seconds=.01))
        now[0] += .01
        runner.tick(t+.01, state)
    hold(3.)
    assert runner.path is None
    now[0] = 11.
    hold(4.)
    assert runner.descent_used and not runner.braking
    assert runner.events[-1]['action']=='observe_down'
    end = runner.path_start+runner.path.duration
    state[:3] = runner.path.sample(runner.path.duration)[0]
    runner.observe(dict(points=np.empty((0,3)), evidence=evidence, position=state[:3].copy(),
                        yaw=0., timestamp=end, goal=np.array([6.,0.,0.])))
    runner.tick(end, state)
    np.testing.assert_allclose(runner.hover, [0., 0., 1.8])
    now[0] = 20.
    hold(end+.1)
    assert runner.path is None
    assert sum(e['action']=='observe_down' for e in runner.events)==1
