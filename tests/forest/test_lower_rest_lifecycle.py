"""Exercise the actual lower loop's recovery/maintenance transition without Isaac."""
import gc
import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from omni_drones.navigation.forest import lower_flight, stable_spline
from omni_drones.navigation.forest.guidance import Guidance
from omni_drones.navigation.forest.subgoal import Subgoal


class StateView:
    def __init__(self, env):
        self.env = env

    def __getitem__(self, _):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.env.vector.copy()


class ClearSafety:
    def clear(self, _):
        return True

    def known(self, *args, **kwargs):
        return True


class HoldPath:
    def sample(self, _):
        return np.zeros((3, 3))

    def as_dict(self):
        return dict(test_hold=True)


@pytest.mark.parametrize('interruption', ['none', 'activation', 'drift'])
def test_recovery_rest_is_reused_only_until_motion_reference_or_measured_rest_changes(
        tmp_path, monkeypatch, interruption):
    cfg = OmegaConf.create(dict(executor='stable_spline', realtime=False, max_seconds=10., history_seconds=3.,
        task=dict(flight=dict(max_speed=1., max_acceleration=1., max_jerk=4., settle_seconds=0.)),
        local=dict(hz=10., lookahead=2., ray_traversal='voxel_dda_v1', connect_start=True,
            heading_max_rate=1., heading_max_acceleration=2., align_before_motion=False,
            max_observation_age=.3),
        lower=dict(map_window=[4., 4., 4.], acceleration_fraction=1., jerk_fraction=1.,
            position_tolerance=.15, velocity_tolerance=.2, stop_dwell=.2, stop_timeout=5.,
            defer_full_gc_while_moving=True, full_gc_interval=2.2),
        spline=dict(activation_lead=.4, candidate_max_age=.8, process_worker=False)))
    goal = Subgoal('inspect', 1, (100., 0., 0.), timeout_seconds=100., no_motion_seconds=100.)
    env = SimpleNamespace(elapsed=0., dt=.1, velocity_control=False, reason=None,
        scene_info=dict(start=[0., 0., 0.], initial_yaw=0.), vector=np.zeros(13))
    env.vector[3] = 1.
    env.state = StateView(env)
    env.set_heading_feedforward = lambda *args: None
    env.scan = lambda: dict(timestamp=env.elapsed)
    clock_offset, stops, failed_once, activation_sent = [0.], [], [False], [False]
    safety = ClearSafety()
    safety.flight = SimpleNamespace(**dict(cfg.task.flight))

    def step(env, velocity, acceleration, yaw, position):
        env.elapsed = round(env.elapsed+env.dt, 8)
        if interruption == 'drift' and env.elapsed >= 1.:
            env.vector[0] = .06
        if env.elapsed >= 3.1:
            env.reason = 'end_test'
        return np.r_[env.elapsed, velocity, acceleration, yaw, env.vector]

    real_start_stop = lower_flight.start_stop
    def start_stop(command, elapsed, *args, **kwargs):
        plan = real_start_stop(command, elapsed, *args, **kwargs)
        stops.append(dict(time=env.elapsed, command=command, plan=plan))
        return plan

    def execution_check(*args, **kwargs):
        reject = not failed_once[0] and env.elapsed == .2
        failed_once[0] |= reject
        return SimpleNamespace(accepted=not reject, reason='REFERENCE_UNOBSERVED' if reject else '',
            nominal_envelope=np.zeros((1, 3)), actual_brake_envelope=np.zeros((1, 3)))

    class Pipeline:
        def __init__(self, *args):
            pass

        def push(self, *args, **kwargs):
            pass

        def poll(self):
            now = env.elapsed
            if 1.9 < now < 2.3:
                return None
            source = 1.868 if now == 1.9 else now
            return dict(points=np.empty((0, 3)), evidence=np.empty((0, 3)), safety=safety,
                planning_safety=safety, state=env.vector.copy(), timestamp=source,
                captured=source+clock_offset[0], goal_id=goal.goal_id, revision=goal.revision,
                decision=dict(raw_velocity=[0., 0., 0.]), compute_seconds=0.,
                target=np.asarray(goal.target_world), distance_field=None, map_origin=np.zeros(3))

    class Coordinator:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, *args):
            return False

        def poll(self, *args):
            if interruption == 'activation' and env.elapsed == 1. and not activation_sent[0]:
                activation_sent[0] = True
                return SimpleNamespace(path=HoldPath(), request=SimpleNamespace(
                    switch_time=env.elapsed, start_pva=np.zeros((3, 3)))), None
            return None, None

    def collect(_):
        clock_offset[0] += .28
        return 0

    monkeypatch.setattr(lower_flight, 'step_velocity', step)
    monkeypatch.setattr(lower_flight, 'start_stop', start_stop)
    monkeypatch.setattr(lower_flight, 'execution_check', execution_check)
    monkeypatch.setattr(lower_flight, 'ObservationPipeline', Pipeline)
    monkeypatch.setattr(stable_spline, 'StableSplinePlanner', Coordinator)
    monkeypatch.setattr(lower_flight.time, 'perf_counter', lambda: env.elapsed+clock_offset[0])
    monkeypatch.setattr(gc, 'collect', collect)
    enabled = gc.isenabled()
    gc.enable()
    try:
        result = lower_flight.run_lower(env, cfg, tmp_path, [],
            Guidance('test', 1, goal.goal_id, np.array([[0., 0., 0.], [100., 0., 0.]])), goal)
    finally:
        if not enabled:
            gc.disable()
    events = [json.loads(line) for line in (tmp_path/'decisions.jsonl').read_text().splitlines()]
    assert result['reason'] == 'end_test'
    assert stops[0]['time'] == .2
    maintenance = next(e for e in events if e['event'] == 'maintenance_start')
    if interruption == 'none':
        assert len(stops) == 1
        assert maintenance['timestamp'] == 2.2
        assert maintenance['original_command_start'] == .2
        rest = next(e for e in events if e['event'] == 'measured_stop')
        assert maintenance['timestamp']-rest['timestamp'] == pytest.approx(1.7)
        complete = next(e for e in events if e['event'] == 'maintenance_complete')
        assert complete['original_command_start'] == .2
        assert complete['post_gc_snapshot_captured'] > complete['gc_end_wall']
    else:
        assert len(stops) == 2
        assert stops[1]['time'] >= 2.3  # Stale data cannot authorize a replacement stop.
        assert maintenance['original_command_start'] == stops[1]['time']
        if interruption == 'activation':
            assert stops[1]['command'] is not stops[0]['plan'].command
