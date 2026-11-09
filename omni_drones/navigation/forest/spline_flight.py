"""Observed Python spline execution with fixed-time asynchronous handoffs.

All planning inputs are measured state, rays and public task bounds. Scene
geometry is deliberately absent; the caller audits the recorded flight later.
"""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import asdict, replace
import json
import time
from types import SimpleNamespace

import numpy as np

from .flight import write_json
from .guidance import GuidanceTracker
from .geometry import yaw_from_quat
from .local import LocalMap, LocalPlanner, VelocityTrajectory
from .local_flight import step_velocity
from .observed_esdf import ObservedESDF
from .spline_optimizer import PythonSplineOptimizer
from .spline_runtime import (ExecutableSpline, SplineCandidate, activation_check, brake_from_reference,
                             execution_check, make_handoff, observation_age)
from .subgoal import GoalMonitor, HeadingReference, angle_error


def anchored_route(route, anchor):
    """Drop route vertices already passed by the future reference anchor."""
    route, anchor = np.asarray(route, float), np.asarray(anchor, float)
    if len(route) < 2:
        return np.array([anchor])
    delta = np.diff(route, axis=0)
    fractions = np.clip(np.sum((anchor-route[:-1])*delta, axis=1)
                        / np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0., 1.)
    nearest = int(np.argmin(np.linalg.norm(route[:-1]+fractions[:, None]*delta-anchor, axis=1)))
    result = np.vstack([anchor, route[nearest+1:]])
    return result[np.r_[True, np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-6]]


def run_spline(env, cfg, output, traces, guidance, goal_spec=None, mission=None):
    if env.velocity_control:
        raise ValueError('Spline execution requires full position/velocity/acceleration control')
    from .esdf import astar
    astar(np.ones((2, 2, 2), bool), np.zeros((2, 2, 2), np.float32),
          np.array([0, 0, 0]), np.array([1, 1, 1]))
    goal_spec = mission.current_goal if mission is not None else goal_spec
    monitor = mission.monitor if mission is not None else GoalMonitor(goal_spec, env.elapsed)
    tracker = GuidanceTracker(guidance, cfg.local.lookahead)
    memory = LocalMap(guidance.points, cfg.history_seconds, ray_traversal=cfg.local.ray_traversal)
    planner = LocalPlanner(cfg.task.flight, cfg.local.hz, connect_start=cfg.local.connect_start)
    planning_flight = SimpleNamespace(**dict(cfg.task.flight))
    planning_flight.max_acceleration *= cfg.spline.acceleration_fraction
    planning_flight.max_jerk *= cfg.spline.jerk_fraction
    optimizer = PythonSplineOptimizer(planning_flight)
    heading = HeadingReference(env.scene_info['initial_yaw'], cfg.local.heading_max_rate,
                               cfg.local.heading_max_acceleration)
    state = env.state[0, 0, :13].detach().cpu().numpy()
    command = VelocityTrajectory(state[:3], np.zeros(3), np.zeros(3), np.zeros(3), cfg.task.flight)
    command_start, generation, braking = env.elapsed, 0, True
    initial_bearing = float(np.arctan2(goal_spec.target_world[1]-state[1], goal_spec.target_world[0]-state[0]))
    aligned = not cfg.local.align_before_motion
    ready, latest, future, optimization_future = None, None, None, None
    pending, audit_frames, acceleration_window = [], deque(), deque(maxlen=5)
    previous_v, previous_t = state[7:10].copy(), env.elapsed
    lease_until = -np.inf
    counters = Counter()
    ages, compute_times, control_times, ref_positions = [], [], [], []
    scan_period = round(1/cfg.local.hz/env.dt)
    horizon = 2/cfg.local.hz
    reason, last_log = 'timeout', None
    begin = time.perf_counter()
    write_json(output/'guidance.json', asdict(guidance) | dict(points=guidance.points.tolist(),
        corridor=guidance.corridor.tolist() if guidance.corridor is not None else None))

    def prepare(frames, measured, actual_a, guide, captured):
        for frame in frames:
            memory.add(frame)
        t = float(frames[-1]['timestamp'])
        points, evidence = memory.snapshot(t, measured[:3])
        _, decision, safety = planner.plan(points, evidence, measured[:3], yaw_from_quat(measured[3:7]),
                                           measured[7:10], actual_a, guide, geometry_only=True)
        return dict(safety=safety, points=points, evidence=evidence, state=measured,
                    timestamp=t, captured=captured, revision=guide.guidance.revision,
                    goal_id=guide.guidance.segment_id, decision=decision,
                    seconds=time.perf_counter()-captured)

    def optimize(snapshot, request, target, speed):
        began = time.perf_counter()
        safety, measured = snapshot['safety'], snapshot['state']
        route = anchored_route(snapshot['decision']['route'], request.start_pva[0])
        candidate, optimization = None, None
        if len(route) > 1:
            target = np.asarray(target)
            if len(route) > 1 and np.linalg.norm(route[-1]-target) < .6:
                if safety.clear([route[-1], target]) and safety.known([route[-1], target], measured[:3]):
                    route = np.vstack([route, target])
            field = ObservedESDF(snapshot['points'], snapshot['evidence'])
            optimized = optimizer.optimize(route, *request.start_pva, field, safety, nominal_speed=speed,
                                            actual_position=measured[:3])
            optimization = dict(validation=asdict(optimized.validation), attempts=optimized.attempts)
            if optimized.path is not None:
                # The optimizer independently validates the full curve using the
                # measured body position and exact future p/v/a. Avoid a second
                # identical full-curve validation in the same worker snapshot.
                candidate = SplineCandidate(request, ExecutableSpline(optimized.path)
                    if optimized.validation.accepted else None, optimized.validation)
        return dict(candidate=candidate, optimization=optimization, seconds=time.perf_counter()-began)

    def log(stream, t, event, **data):
        stream.write(json.dumps(dict(timestamp=t, event=event, generation=generation,
            goal_id=goal_spec.goal_id, revision=goal_spec.revision, **data))+'\n')

    with ThreadPoolExecutor(max_workers=1) as pool, ThreadPoolExecutor(max_workers=1) as optimizer_pool, (output/'decisions.jsonl').open('w') as stream:
        for step in range(round(cfg.max_seconds/env.dt)):
            clock, t = time.perf_counter(), env.elapsed
            state = env.state[0, 0, :13].detach().cpu().numpy()
            yaw = yaw_from_quat(state[3:7])
            if t > previous_t+1e-8:
                acceleration_window.append((state[7:10]-previous_v)/(t-previous_t))
            actual_a = np.mean(acceleration_window, axis=0) if acceleration_window else np.zeros(3)
            previous_v, previous_t = state[7:10].copy(), t
            tracker.update(state[:3], yaw, t)
            p, v, a = command.sample(max(0., t-command_start))
            if not aligned and t >= cfg.task.flight.settle_seconds:
                aligned = abs(angle_error(initial_bearing, yaw)) < .15 and abs(state[12]) < .15
            feedback = (mission.update(t, state[:3], state[7:10], yaw, env.reason) if mission is not None
                        else monitor.update(t, state[:3], state[7:10], yaw, env.reason))
            if mission is not None and mission.status == 'WAITING_NEXT':
                activated = mission.advance(t, state[:3], state[7:10], yaw,
                    reference_settled=np.linalg.norm(v) < .03 and np.linalg.norm(a) < .05)
                if activated is not None:
                    goal_spec, monitor = activated, mission.monitor
                    guidance = replace(tracker.guidance, revision=activated.revision,
                        segment_id=activated.goal_id, points=np.array([state[:3], activated.target_world]), source_time=t)
                    tracker.set(guidance, t)
                    tracker.update(state[:3], yaw, t)
                    initial_bearing = float(np.arctan2(activated.target_world[1]-state[1], activated.target_world[0]-state[0]))
                    aligned, ready = not cfg.local.align_before_motion, None
                    generation += 1
                    log(stream, t, 'goal_activated')
                    print(f'SPLINE_NEXT t={t:.2f} goal={activated.goal_id}', flush=True)
            if mission is not None and mission.status in ('COMPLETE', 'FAILED'):
                reason = mission.reason
                break
            if mission is None and feedback['status'] != 'RUNNING':
                reason = feedback['reason']
                break
            checks = dict(actual_position=state[:3], actual_velocity=state[7:10], actual_acceleration=actual_a,
                          horizon=horizon, position_tolerance=cfg.spline.position_tolerance,
                          velocity_tolerance=cfg.spline.velocity_tolerance)
            near_goal = np.linalg.norm(state[:3]-goal_spec.target_world) <= goal_spec.position_tolerance
            holding = not aligned or near_goal or (mission is not None and mission.status == 'WAITING_NEXT')
            if optimization_future is not None and optimization_future.done():
                optimized, optimization_future = optimization_future.result(), None
                ready = optimized['candidate']
                counters['candidate_rejected'] += int(ready is None)
                log(stream, t, 'optimization', optimization=optimized['optimization'],
                    compute_seconds=optimized['seconds'])
            if future is not None and future.done():
                result, future = future.result(), None
                latest = result
                age = observation_age(t, time.perf_counter(), result['timestamp'], result['captured'])
                ages.append(age); compute_times.append(result['seconds'])
                current = (result['goal_id'] == goal_spec.goal_id and result['revision'] == goal_spec.revision)
                valid = age <= cfg.local.max_observation_age
                if valid:
                    checked = execution_check(command, max(0., t-command_start), result['safety'], **checks)
                    if checked.accepted:
                        lease_until = t+horizon
                    else:
                        lease_until = -np.inf
                else:
                    checked = None
                if valid and current and not holding and optimization_future is None and ready is None:
                    request = make_handoff(command, command_start, t, result['timestamp'], result['captured'],
                        generation, goal_spec.revision, activation_lead=cfg.spline.activation_lead, control_dt=env.dt)
                    optimization_future = optimizer_pool.submit(optimize, result, request,
                        goal_spec.target_world, tracker.guidance.speed_limit)
                log(stream, t, 'observation', source_time=result['timestamp'], pipeline_seconds=age,
                    compute_seconds=result['seconds'], current_goal=current,
                    tracking_check=checked.reason if checked else 'STALE',
                    actual_position=state[:3].tolist(), actual_velocity=state[7:10].tolist(),
                    actual_acceleration=actual_a.tolist())
            reject = ''
            if ready is not None:
                checked = activation_check(ready, command, command_start, t, time.perf_counter(), generation,
                    goal_spec.revision, latest['safety'], deadline=cfg.local.deadline,
                    max_observation_age=cfg.local.max_observation_age, **checks)
                if checked.accepted and not holding:
                    command, command_start = ready.path, ready.request.switch_time
                    generation += 1
                    braking, lease_until = False, t+horizon
                    counters['spline_activations'] += 1
                    log(stream, t, 'spline_activated', switch_time=command_start,
                        source_time=ready.request.source_time, start_pva=ready.request.start_pva.tolist(),
                        actual_position=state[:3].tolist(), actual_velocity=state[7:10].tolist(),
                        position_error=checked.position_error, velocity_error=checked.velocity_error,
                        spline=command.as_dict())
                    ready = None
                elif checked.reason != 'WAIT' or holding:
                    reject = checked.reason or 'HOLD'
                    counters['rejected_'+reject] += 1
                    log(stream, t, 'candidate_rejected', reason=reject)
                    ready = None
            stale = latest is None or observation_age(t, time.perf_counter(), latest['timestamp'], latest['captured']) > cfg.local.max_observation_age
            if not braking and (holding or stale or t > lease_until+1e-8):
                command, checked = brake_from_reference(command, max(0., t-command_start), latest['safety'] if latest else None,
                    actual_position=state[:3], actual_velocity=state[7:10], actual_acceleration=actual_a,
                    flight=cfg.task.flight)
                command_start, braking, ready = t, True, None
                generation += 1
                counters['brakes'] += 1
                counters['uncertified_brakes'] += int(not checked.accepted)
                log(stream, t, 'brake', reason='HOLD' if holding else 'STALE' if stale else 'LEASE_EXPIRED',
                    certified=checked.accepted, check_reason=checked.reason)
                if not checked.accepted:
                    reason = 'uncertified_stop'
                    break
            if step % scan_period == 0:
                captured = time.perf_counter()
                frame = env.scan()
                pending.append(frame); audit_frames.append(frame)
                while audit_frames and float(audit_frames[0]['timestamp']) < t-cfg.history_seconds-1.:
                    audit_frames.popleft()
                if future is None and t >= cfg.task.flight.settle_seconds:
                    future = pool.submit(prepare, pending, state.copy(), actual_a.copy(), copy(tracker), captured)
                    pending = []
            # PhysX uses a float32 step; an accepted tick may precede the
            # mathematical switch by a few nanoseconds within clock_tolerance.
            p, v, a = command.sample(max(0., t-command_start))
            target_yaw = (goal_spec.arrival_yaw if np.linalg.norm(state[:3]-goal_spec.target_world) < 1.
                and goal_spec.arrival_yaw is not None else np.arctan2(v[1], v[0]) if np.linalg.norm(v[:2]) > .08 else heading.yaw)
            if not aligned and t >= cfg.task.flight.settle_seconds:
                target_yaw = initial_bearing
            reference_yaw, rate, yaw_a = heading.step(target_yaw, env.dt)
            env.set_heading_feedforward(rate, yaw_a)
            ref_positions.append(p)
            traces.append(step_velocity(env, v, a, reference_yaw, p))
            if env.reason:
                reason = env.reason
                break
            control_times.append(time.perf_counter()-clock)
            status = (goal_spec.goal_id, braking, reject)
            if status != last_log:
                print(f'SPLINE t={t:.2f} goal={goal_spec.goal_id} braking={braking} reject={reject}', flush=True)
                last_log = status
            if cfg.realtime:
                time.sleep(max(0., begin+(step+1)*env.dt-time.perf_counter()))
    state = env.state[0, 0, :13].detach().cpu().numpy()
    if latest is not None:
        np.savez_compressed(output/'last_observation.npz', points=latest['points'], evidence=latest['evidence'],
            position=latest['state'][:3], yaw=yaw_from_quat(latest['state'][3:7]), timestamp=latest['timestamp'])
    if audit_frames:
        folder = output/'observation_window'
        folder.mkdir()
        for index, frame in enumerate(audit_frames):
            np.savez_compressed(folder/f'{index:04d}.npz', **frame)
        write_json(folder/'manifest.json', dict(frames=len(audit_frames),
            first_timestamp=float(audit_frames[0]['timestamp']), last_timestamp=float(audit_frames[-1]['timestamp']),
            ray_traversal=memory.ray_traversal, map_origin=memory.origin.tolist(),
            scope='last history_seconds + 1 second; not the full flight'))
    np.save(output/'reference_positions.npy', np.asarray(ref_positions).reshape(-1, 3))
    summary = dict(reason=reason, **counters,
        goal_distance=float(np.linalg.norm(state[:3]-goal_spec.target_world)),
        pipeline_p95_seconds=float(np.percentile(ages, 95)) if ages else None,
        compute_p95_seconds=float(np.percentile(compute_times, 95)) if compute_times else None,
        control_p95_seconds=float(np.percentile(control_times, 95)) if control_times else None,
        control_overruns=sum(t > env.dt for t in control_times))
    summary['subgoal'] = monitor.update(env.elapsed, state[:3], state[7:10], yaw_from_quat(state[3:7]),
        None if reason in ('goal_reached', 'mission_reached') else reason)
    if mission is not None:
        if mission.status not in ('COMPLETE', 'FAILED'):
            mission.update(env.elapsed, state[:3], state[7:10], yaw_from_quat(state[3:7]), reason)
        summary['mission'] = mission.feedback()
        write_json(output/'mission_events.json', mission.events)
        write_json(output/'goal_results.json', mission.results)
    if len(traces):
        errors = np.linalg.norm(np.asarray(traces)[:, 8:11]-ref_positions, axis=1)
        summary['position_reference_error_m'] = dict(mean=float(errors.mean()), maximum=float(errors.max()))
    return summary
