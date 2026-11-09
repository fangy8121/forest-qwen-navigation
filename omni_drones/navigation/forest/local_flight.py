"""阶段 A：Lee 速度/制动验证与固定任务引导下的主动单行飞行。"""
import json
import time
from copy import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from collections import deque
from pathlib import Path
import numpy as np
from .flight import new_run, write_json
from .geometry import (local_to_world, yaw_from_quat, GRID_RESOLUTION, EVIDENCE_TOLERANCE,
                       INFLATION, VERTICAL_RADIUS, observed_body_height)
from .guidance import Guidance, GuidanceTracker
from .local import LocalMap, LocalPlanner, VelocityTrajectory, certificate, BODY_NUMERIC_TOLERANCE
from .runs import file_hash
from .trajectory import QuinticPath


def local_protocol(cfg):
    return dict(name='forest_local_v1', guidance_coordinates='world_xyz',
        action=('velocity_world(3), acceleration_world(3), yaw_world' if cfg.task.get('control_mode', 'trajectory') == 'velocity'
                else 'position_world(3), velocity_world(3), acceleration_world(3), yaw_world'),
        control_hz=1/float(cfg.sim.dt), local_hz=float(cfg.local.hz),
        execution_horizon_seconds=2/float(cfg.local.hz), brake='complete',
        deadline_seconds=float(cfg.local.deadline), max_observation_age_seconds=float(cfg.local.max_observation_age),
        history_seconds=float(cfg.history_seconds), evidence_resolution_m=GRID_RESOLUTION,
        free_space_traversal=cfg.local.get('ray_traversal', 'sampled_v1'),
        evidence_tolerance_m=float(EVIDENCE_TOLERANCE), current_body_padding_m=BODY_NUMERIC_TOLERANCE,
        collision_horizontal_m=INFLATION, collision_vertical_m=VERTICAL_RADIUS,
        observed_body_half_height_m=observed_body_height(cfg.task.flight.max_acceleration),
        hidden_geometry_input=False, teacher_path_input=False,
        task_guidance='public start/goal and corridor', observation_action='certified vertical probe')


def step_velocity(env, velocity, acceleration, yaw, position=None):
    import torch
    from tensordict import TensorDict
    command = np.r_[velocity, acceleration, yaw]
    if not env.velocity_control and position is None:
        raise ValueError('Position-controlled execution requires a position reference')
    control = command if env.velocity_control else np.r_[position, command]
    action = torch.as_tensor(control, dtype=torch.float32, device=env.device).reshape(1, 1, -1)
    env.step(TensorDict({'agents': {'action': action}}, [1]))
    if (env.enable_viewport and env.cfg.viewer.get('follow_drone', False)
            and not getattr(env, '_camera_updates_with_render', False)):
        env.update_camera()
    state = env.state[0, 0, :13].detach().cpu().numpy()
    return np.r_[env.elapsed, command, state]


def run_control(env, cfg, traces):
    phases, step_times = [], []
    commands = [('forward', [.6, 0., 0.]), ('left', [0., .6, 0.]),
                ('up', [0., 0., .4]), ('down', [0., 0., -.4])]
    for yaw in (0., np.pi/2):
        env.scene_info['initial_yaw'] = yaw
        env.reset()
        for _ in range(round(3./env.dt)):
            traces.append(step_velocity(env, np.zeros(3), np.zeros(3), yaw))
        for name, local_velocity in commands:
            target = local_to_world([local_velocity], np.zeros(3), yaw)[0]
            for moving in (True, False):
                state = env.state[0, 0, :13].detach().cpu().numpy()
                desired = target if moving else np.zeros(3)
                path = VelocityTrajectory(state[:3], state[7:10], np.zeros(3), desired, cfg.task.flight)
                begin = state[:3].copy()
                duration = 2.5 if moving else 3.
                rows = []
                for index in range(round(duration/env.dt)):
                    clock = time.perf_counter()
                    _, velocity, acceleration = path.sample(index*env.dt)
                    row = step_velocity(env, velocity, acceleration, yaw)
                    traces.append(row); rows.append(row)
                    step_times.append(time.perf_counter()-clock)
                    if env.reason:
                        return dict(reason=env.reason, phases=phases)
                rows = np.asarray(rows)
                error = np.linalg.norm(rows[-round(.5/env.dt):, 15:18]-desired, axis=1)
                measured = float(np.linalg.norm(rows[:, 8:11]-begin, axis=1).max())
                predicted = float(np.linalg.norm(path.sample(duration)[0]-begin))
                phases.append(dict(yaw=yaw, phase=name if moving else 'brake_'+name,
                    target_velocity_world=desired.tolist(), plateau_velocity_error=float(error.mean()),
                    distance=measured, predicted_distance=predicted,
                    passed=bool(error.mean() < .08 and (moving or measured <= predicted+.12))))
    return dict(reason='control_passed' if all(p['passed'] for p in phases) else 'control_failed',
                phases=phases, control_p95_seconds=float(np.percentile(step_times, 95)))


def run_baseline(env, cfg, output, traces, guidance=None, goal_spec=None, mission=None):
    from .esdf import astar
    # JIT 属于启动开销，在控制回路开始前完成。
    astar(np.ones((2, 2, 2), bool), np.zeros((2, 2, 2), np.float32),
          np.array([0, 0, 0]), np.array([1, 1, 1]))
    if mission is not None:
        goal_spec = mission.current_goal
    start = np.asarray(env.scene_info['start'])
    goal = np.asarray(goal_spec.target_world if goal_spec else env.scene_info['goal'])
    if guidance is None:
        bounds = env.scene_info['validation']['route_bounds']
        corridor = np.array([[bounds[0], bounds[2]], [bounds[1], bounds[2]],
                             [bounds[1], bounds[3]], [bounds[0], bounds[3]]])
        guidance = Guidance('single_row', 1, 'row_0', np.array([start, goal]),
                        speed_limit=min(cfg.local.speed_limit, cfg.task.flight.max_speed), corridor=corridor,
                        height_bounds=(float(min(start[2], goal[2])-.8), float(max(start[2], goal[2])+.8)))
    if goal_spec is not None:
        if (guidance.segment_id != goal_spec.goal_id or guidance.revision != goal_spec.revision
                or not np.allclose(guidance.points[-1], goal)):
            raise ValueError('guidance and subgoal identity/target must match')
        from .subgoal import GoalMonitor, HeadingReference, angle_error
        monitor = mission.monitor if mission is not None else GoalMonitor(goal_spec, env.elapsed)
        heading = HeadingReference(env.scene_info['initial_yaw'],
            min(cfg.task.flight.max_yaw_rate, cfg.local.get('heading_max_rate', cfg.task.flight.max_yaw_rate)),
            min(cfg.task.flight.max_yaw_acceleration,
                cfg.local.get('heading_max_acceleration', cfg.task.flight.max_yaw_acceleration)))
        initial_bearing = float(np.arctan2(goal[1]-start[1], goal[0]-start[0]))
        aligned = not cfg.local.get('align_before_motion', False)
    tracker = GuidanceTracker(guidance, cfg.local.lookahead)
    write_json(output/'guidance.json', dict(asdict(guidance), points=guidance.points.tolist(),
                                           corridor=guidance.corridor.tolist() if guidance.corridor is not None else None))
    memory = LocalMap(guidance.points, cfg.history_seconds,
                      ray_traversal=cfg.local.get('ray_traversal', 'sampled_v1'))
    planner = LocalPlanner(cfg.task.flight, cfg.local.hz,
                           connect_start=cfg.local.get('connect_start', False))

    def prepare(frames, state, acceleration, guide, captured):
        for frame in frames:
            memory.add(frame)
        frame = frames[-1]
        t = float(frame['timestamp'])
        if t < cfg.task.flight.settle_seconds:
            return None
        points, evidence = memory.snapshot(t, state[:3])
        path, decision, safety = planner.plan(points, evidence, state[:3], yaw_from_quat(state[3:7]),
                                              state[7:10], acceleration, guide)
        return dict(path=path, decision=decision, safety=safety, points=points, evidence=evidence,
                    state=state, timestamp=t, captured=captured, revision=guide.guidance.revision,
                    goal_id=guide.guidance.segment_id,
                    compute_seconds=time.perf_counter()-captured)

    def brake(state, velocity, acceleration, safety, reference_position=None):
        origin = state[:3] if env.velocity_control else reference_position
        path = QuinticPath.brake(origin, velocity, acceleration,
                                cfg.task.flight.max_acceleration, cfg.task.flight.max_jerk)
        envelope = path.remaining(0., state[:3])
        actual = QuinticPath.brake(state[:3], state[7:10], acceleration,
                                  cfg.task.flight.max_acceleration, cfg.task.flight.max_jerk)
        envelope = np.vstack([envelope, actual.remaining(0., state[:3])])
        certified = safety is not None and safety.clear(envelope) and safety.known(envelope, state[:3], body_padding=BODY_NUMERIC_TOLERANCE)
        return path, bool(certified), envelope

    state = env.state[0, 0, :13].detach().cpu().numpy()
    command = VelocityTrajectory(state[:3], state[7:10], np.zeros(3), np.zeros(3), cfg.task.flight)
    command_start, braking = env.elapsed, True
    timings, compute_times, feedbacks, applied_times, step_times = [], [], [], [], []
    reference_positions = []
    audit_frames = deque()
    pending, future, latest = [], None, None
    last_reason, reason = None, 'timeout'
    certified_steps, shield_steps, deadline_misses, control_overruns = 0, 0, 0, 0
    scan_period = round(1/cfg.local.hz/env.dt)
    loop_begin = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool, (output/'decisions.jsonl').open('w') as decisions:
        for step in range(round(cfg.max_seconds/env.dt)):
            clock = time.perf_counter()
            t = env.elapsed
            state = env.state[0, 0, :13].detach().cpu().numpy()
            yaw = yaw_from_quat(state[3:7])
            if goal_spec is not None and not aligned and t >= cfg.task.flight.settle_seconds:
                aligned = abs(angle_error(initial_bearing, yaw)) < .15 and abs(state[12]) < .15
            tracker.update(state[:3], yaw, t)
            reference_p, reference_v, reference_a = command.sample(t-command_start)
            if mission is not None:
                mission_feedback = mission.update(t, state[:3], state[7:10], yaw, env.reason)
                if mission_feedback['status'] == 'WAITING_NEXT':
                    if not braking:
                        command, _, _ = brake(state, reference_v, reference_a,
                            latest['safety'] if latest else None, reference_p)
                        command_start, braking = t, True
                        reference_p, reference_v, reference_a = command.sample(0.)
                    activated = mission.advance(t, state[:3], state[7:10], yaw,
                        reference_settled=(np.linalg.norm(reference_v) < .03
                                           and np.linalg.norm(reference_a) < .05))
                    if activated is not None:
                        goal_spec, goal, monitor = activated, np.asarray(activated.target_world), mission.monitor
                        guidance = replace(tracker.guidance, revision=activated.revision,
                            segment_id=activated.goal_id, points=np.array([state[:3], goal]), source_time=t)
                        tracker.set(guidance, t)
                        tracker.update(state[:3], yaw, t)
                        initial_bearing = float(np.arctan2(goal[1]-state[1], goal[0]-state[0]))
                        aligned = not cfg.local.get('align_before_motion', False)
                        # Only display metadata changes: no physical state or clock reset.
                        env.scene_info['goal'] = goal.tolist()
                        write_json(output/'mission_events.json', mission.events)
                        print(f'MISSION_NEXT t={t:.2f} goal={activated.goal_id} revision={activated.revision}', flush=True)
                if mission.status in ('COMPLETE', 'FAILED'):
                    reason = mission.reason
                    break
            if future is not None and future.done():
                result = future.result()
                future = None
                # An old worker may still finish integrating scans after a handoff.
                # Keep its map updates, but never execute or brake for its old goal.
                if result is not None and mission is not None and not mission.accepts(
                        result['goal_id'], result['revision']):
                    result = None
                if result is not None:
                    latest = result
                    decision, safety = result['decision'], result['safety']
                    if goal_spec is not None and not aligned:
                        decision = dict(decision, executed_velocity=[0., 0., 0.], action='ALIGN')
                    if goal_spec is not None and np.linalg.norm(state[:3]-goal) <= goal_spec.position_tolerance:
                        decision = dict(decision, executed_velocity=[0., 0., 0.], action='SETTLE')
                    age = max(t-result['timestamp'], time.perf_counter()-result['captured'])
                    late = age > cfg.local.deadline or result['revision'] != tracker.guidance.revision
                    if not late:
                        origin = state[:3] if env.velocity_control else reference_p
                        command = VelocityTrajectory(origin, reference_v, reference_a,
                                                     decision['executed_velocity'], cfg.task.flight)
                        rejection, envelope = certificate(safety, command, state[:3], 2/cfg.local.hz)
                    else:
                        rejection = 'STALE'
                    if rejection:
                        command, certified, envelope = brake(state, reference_v, reference_a, safety, reference_p)
                        decision.update(executed_velocity=[0., 0., 0.], reason=rejection,
                                        certified=certified, shield_intervened=True, action='BRAKE')
                    else:
                        decision['certified'] = True
                    age = max(t-result['timestamp'], time.perf_counter()-result['captured'])
                    if age > cfg.local.deadline and not late:
                        command, certified, envelope = brake(state, reference_v, reference_a, safety, reference_p)
                        decision.update(executed_velocity=[0., 0., 0.], reason='STALE',
                                        certified=certified, shield_intervened=True, action='BRAKE')
                        rejection, late = 'STALE', True
                        age = max(t-result['timestamp'], time.perf_counter()-result['captured'])
                    decision['certified'] &= age <= cfg.local.max_observation_age
                    decision['envelope'] = envelope.tolist()
                    timings.append(age); compute_times.append(result['compute_seconds'])
                    applied_times.append(time.perf_counter())
                    command_start, braking = t, bool(rejection)
                    certified_steps += int(decision['certified'])
                    shield_steps += int(decision['shield_intervened'])
                    deadline_misses += int(late)
                    status = 'RUNNING' if not decision['reason'] else 'BLOCKED'
                    feedback = asdict(tracker.feedback(t, status, decision['reason'], age))
                    feedbacks.append(feedback)
                    decisions.write(json.dumps(dict(timestamp=t, source_time=result['timestamp'],
                        pipeline_seconds=age, feedback=feedback, actual_position=state[:3].tolist(),
                        actual_velocity=state[7:10].tolist(), reference_position=reference_p.tolist(),
                        reference_velocity=reference_v.tolist(), reference_acceleration=reference_a.tolist(), **decision))+'\n')
                    if decision['reason'] != last_reason:
                        print(f'LOCAL t={t:.1f} progress={tracker.progress:.2f} '
                              f'reason={decision["reason"] or "RUNNING"} age={age:.3f}s', flush=True)
                        last_reason = decision['reason']
                    if not decision['certified'] and np.linalg.norm(state[7:10]) < cfg.task.flight.settle_speed:
                        reason = 'uncertified_stop'; break
            if step % scan_period == 0:
                captured = time.perf_counter()
                frame = env.scan()
                pending.append(frame)
                if cfg.local.get('save_observation_window', False):
                    audit_frames.append(frame)
                    while audit_frames and float(audit_frames[0]['timestamp']) < t-cfg.history_seconds-1.:
                        audit_frames.popleft()
                if future is None:
                    future = pool.submit(prepare, pending, state.copy(), reference_a.copy(), copy(tracker), captured)
                    pending = []
            if goal_spec is None and t >= cfg.task.flight.settle_seconds and np.linalg.norm(state[:3]-goal) < .3:
                if not braking:
                    command, _, _ = brake(state, reference_v, reference_a,
                                          latest['safety'] if latest else None, reference_p)
                    command_start, braking = t, True
                if np.linalg.norm(state[7:10]) < cfg.task.flight.settle_speed:
                    reason = 'goal_reached'; break
            if not braking and (t-command_start > 2/cfg.local.hz
                    or time.perf_counter()-latest['captured'] > cfg.local.max_observation_age):
                command, _, _ = brake(state, reference_v, reference_a, latest['safety'], reference_p)
                command_start, braking = t, True
                deadline_misses += 1
            if goal_spec is None and t-tracker.last_progress > cfg.local.no_progress_seconds:
                reason = 'no_progress'; break
            reference_position, velocity, acceleration = command.sample(t-command_start)
            reference_yaw = env.scene_info['initial_yaw']
            if goal_spec is not None:
                feedback = monitor.update(t, state[:3], state[7:10], yaw, env.reason)
                if mission is None and feedback['status'] != 'RUNNING':
                    reason = feedback['reason']; break
                target_yaw = (goal_spec.arrival_yaw if np.linalg.norm(state[:3]-goal) < 1.
                              and goal_spec.arrival_yaw is not None else
                              np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity[:2]) > .08
                              else heading.yaw)
                if not aligned and t >= cfg.task.flight.settle_seconds:
                    target_yaw = initial_bearing
                reference_yaw, yaw_rate, yaw_acceleration = heading.step(target_yaw, env.dt)
                env.set_heading_feedforward(yaw_rate, yaw_acceleration)
            reference_positions.append(reference_position)
            traces.append(step_velocity(env, velocity, acceleration, reference_yaw, reference_position))
            if env.reason:
                reason = env.reason; break
            spent = time.perf_counter()-clock
            step_times.append(spent)
            control_overruns += int(spent > env.dt)
            if cfg.realtime:
                time.sleep(max(0., loop_begin+(step+1)*env.dt-time.perf_counter()))
    if latest is not None:
        np.savez_compressed(output/'last_observation.npz', points=latest['points'], evidence=latest['evidence'],
                            position=latest['state'][:3], yaw=yaw_from_quat(latest['state'][3:7]),
                            timestamp=latest['timestamp'])
    if audit_frames:
        folder = output/'observation_window'
        folder.mkdir()
        for i, frame in enumerate(audit_frames):
            np.savez_compressed(folder/f'{i:04d}.npz', **frame)
        write_json(folder/'manifest.json', dict(scope='last history_seconds + 1 second; not the full flight',
            frames=len(audit_frames), first_timestamp=float(audit_frames[0]['timestamp']),
            last_timestamp=float(audit_frames[-1]['timestamp']),
            planner_snapshot_timestamp=float(latest['timestamp']) if latest else None,
            map_origin=memory.origin.tolist(), ray_traversal=memory.ray_traversal))
    intervals = np.diff(applied_times)
    np.save(output/'reference_positions.npy', np.asarray(reference_positions).reshape(-1, 3))
    state = env.state[0, 0, :13].detach().cpu().numpy()
    tracker.update(state[:3], yaw_from_quat(state[3:7]), env.elapsed)
    final_feedback = asdict(tracker.feedback(env.elapsed,
        'COMPLETE' if reason in ('goal_reached', 'mission_reached') else 'BLOCKED', reason,
        time.perf_counter()-latest['captured'] if latest else 0.))
    summary = dict(reason=reason, progress=tracker.progress, route_length=float(tracker.arc[-1]),
        goal_distance=float(np.linalg.norm(env.state[0, 0, :3].detach().cpu().numpy()-goal)),
        decisions=len(timings), certified_steps=certified_steps, shield_steps=shield_steps,
        deadline_misses=deadline_misses, control_overruns=control_overruns,
        pipeline_p50_seconds=float(np.percentile(timings, 50)) if timings else None,
        pipeline_p95_seconds=float(np.percentile(timings, 95)) if timings else None,
        pipeline_max_seconds=max(timings, default=None),
        compute_p95_seconds=float(np.percentile(compute_times, 95)) if compute_times else None,
        control_p95_seconds=float(np.percentile(step_times, 95)) if step_times else None,
        local_output_hz=float(1/intervals.mean()) if len(intervals) else None,
        feedback=final_feedback)
    if goal_spec is not None:
        summary['subgoal'] = monitor.update(env.elapsed, state[:3], state[7:10], yaw_from_quat(state[3:7]),
                                            None if reason == 'goal_reached' else reason)
    if mission is not None:
        if mission.status not in ('COMPLETE', 'FAILED'):
            mission.update(env.elapsed, state[:3], state[7:10], yaw_from_quat(state[3:7]), reason)
        summary['mission'] = mission.feedback()
        write_json(output/'mission_events.json', mission.events)
        write_json(output/'goal_results.json', mission.results)
    if len(traces):
        error = np.linalg.norm(np.asarray(traces)[:, 8:11]-reference_positions, axis=1)
        summary['position_reference_error_m'] = dict(mean=float(error.mean()), maximum=float(error.max()))
    return summary


def run_local(cfg):
    from omegaconf import OmegaConf
    from omni_drones import init_simulation_app
    if cfg.mode == 'control':
        cfg.task.scene.source = 'empty'
    output = new_run(cfg.output)
    write_json(output/'config.json', OmegaConf.to_container(cfg, resolve=True))
    write_json(output/'protocol.json', local_protocol(cfg))
    root = Path(__file__).resolve().parents[3]
    sources = ['cfg/forest_local.yaml', 'cfg/task/ForestSurvey.yaml', 'scripts/fly_forest_local.py',
               'omni_drones/envs/single/forest_survey.py']
    sources += [f'omni_drones/navigation/forest/{name}.py' for name in
                ('guidance', 'local', 'local_flight', 'geometry', 'planning', 'observation', 'trajectory', 'esdf', 'scene')]
    write_json(output/'source_hashes.json', {name: file_hash(root/name) for name in sources})
    print(f'FOREST_OUTPUT={output}', flush=True)
    app = init_simulation_app(cfg)
    env, traces, summary = None, [], dict(reason='initialization_failure')
    began = time.perf_counter()
    try:
        from omni_drones.envs.single.forest_survey import ForestSurvey
        env = ForestSurvey(cfg, cfg.headless)
        env.reset()
        write_json(output/'vehicle.json', dict(dimensions=env.dimensions, physx=env.physics_mass))
        write_json(output/'scene_truth.json', env.scene_info)
        np.savez_compressed(output/'scene_mesh.npz', vertices=env.mesh_vertices, faces=env.mesh_faces)
        summary = run_control(env, cfg, traces) if cfg.mode == 'control' else run_baseline(env, cfg, output, traces)
        if cfg.mode == 'baseline' and traces:
            from .evaluation import truth_collision
            truth = dict(mesh=(env.mesh_vertices, env.mesh_faces))
            actual_path = np.asarray(traces)[:, 8:11]
            summary['truth_margin_violation'] = bool(truth_collision(actual_path[::2], truth,
                                                                    INFLATION, VERTICAL_RADIUS))
            from .esdf import gate_crossings
            gates = [g for g in env.scene_info['validation'].get('task_gates', []) if g['kind'] == 'center']
            covered = [gate_crossings(actual_path, [g]) is not None for g in gates]
            summary.update(required_inspection_gates=len(gates), covered_inspection_gates=sum(covered),
                           inspection_gate_coverage=covered,
                           single_row_complete=summary['reason'] == 'goal_reached' and all(covered))
    except Exception as error:
        summary = dict(reason=type(error).__name__, error=str(error))
        raise
    finally:
        rows = np.asarray(traces)
        np.save(output/'trajectory.npy', rows)
        if len(rows):
            summary.update(actual_max_speed=float(np.linalg.norm(rows[:, 15:18], axis=1).max()),
                reference_max_speed=float(np.linalg.norm(rows[:, 1:4], axis=1).max()),
                reference_max_acceleration=float(np.linalg.norm(rows[:, 4:7], axis=1).max()),
                velocity_error_p95=float(np.percentile(np.linalg.norm(rows[:, 15:18]-rows[:, 1:4], axis=1), 95)))
        summary.update(steps=len(traces), wall_seconds=time.perf_counter()-began,
                       trajectory_columns='t, reference_velocity_world(3), reference_acceleration_world(3), yaw, '
                                          'actual_position_world(3), quaternion_wxyz(4), actual_velocity_world(3), '
                                          'actual_angular_velocity_world(3)')
        write_json(output/'result.json', summary)
        print(f'FOREST_RESULT={summary["reason"]} {output}', flush=True)
        if env is not None:
            env.close()
        app.close()
    return output
