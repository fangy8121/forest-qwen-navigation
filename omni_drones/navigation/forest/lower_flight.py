"""Executable observed local navigation, with a latched measured-stop lifecycle.

Public coordinate goals are the only mission inputs. Neither a truth mesh nor a
teacher route is available here. A failed certificate remains a failed episode;
the controller nevertheless keeps executing one stop reference until a measured
stop (or a separately reported unrecoverable failure/stop timeout).
"""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import time
from types import SimpleNamespace

import numpy as np

from .flight import write_json


def _profile_top(profiler, limit=15):
    """Largest cumulative-time functions of one profiled call, for the decisions log."""
    import pstats
    stats = pstats.Stats(profiler)
    rows = sorted(stats.stats.items(), key=lambda item: -item[1][3])[:limit]
    return [dict(function=f'{name[0].rsplit("/", 1)[-1]}:{name[1]}:{name[2]}', calls=int(calls),
                 cumulative_seconds=round(float(cumulative), 4)) for name, (_, calls, _, cumulative, _) in rows]
from .geometry import yaw_from_quat
from .guidance import GuidanceTracker
from .local import LocalMap, LocalPlanner, VelocityTrajectory, certificate
from .local_flight import step_velocity
from .lower_execution import ActualAccelerationEstimator, MeasuredStopMonitor, start_stop
from .lower_guidance import gate_target, velocity_candidates
from .observed_esdf import ObservedESDF
from .planning import HeightBand
from .spline_runtime import execution_check, observation_age
from .observation_pipeline import ObservationPipeline, observation_stop_cause, stale_stop_uncertified
from .subgoal import GoalMonitor, HeadingReference, angle_error
from .trace_journal import TraceJournal


def run_lower(env, cfg, output, traces, guidance, goal_spec=None, mission=None):
    spline_mode = cfg.get('executor') == 'stable_spline'
    if spline_mode and env.velocity_control:
        raise ValueError('Stable spline tracking requires explicit p/v/a position control')
    if not spline_mode and not env.velocity_control:
        raise ValueError('Executable lower layer currently requires velocity control')
    from .esdf import astar
    astar(np.ones((2, 2, 2), bool), np.zeros((2, 2, 2), np.float32),
          np.array([0, 0, 0]), np.array([1, 1, 1]))
    goal_spec = mission.current_goal if mission is not None else goal_spec
    if goal_spec is None:
        raise ValueError('A public subgoal or mission is required')
    if mission is not None:
        guidance = replace(guidance, points=mission.guidance_points(env.scene_info['start']))
    monitor = mission.monitor if mission is not None else GoalMonitor(goal_spec, env.elapsed)
    tracker = GuidanceTracker(guidance, cfg.local.lookahead)
    memory = LocalMap(guidance.points, cfg.history_seconds, ray_traversal=cfg.local.ray_traversal,
                      rolling=True, window_size=tuple(cfg.lower.map_window),
                      memory_seconds=cfg.local.get('memory_seconds'),
                      evidence_range=float(cfg.local.get('evidence_range', 6.)))
    planner = LocalPlanner(cfg.task.flight, cfg.local.hz, connect_start=cfg.local.connect_start,
        route_persistence=cfg.local.get('route_persistence', False),
        side_commitment=cfg.local.get('side_commitment', False),
        grid_heading=cfg.local.get('grid_heading', 'yaw'),
        max_climb_slope=cfg.local.get('max_climb_slope'), grid_forward=cfg.local.get('grid_forward', 3.2),
        route_refresh_remaining=cfg.local.get('route_refresh_remaining'), agl_band=cfg.local.get('agl_band'))
    # The mission's own ground source (known terrain as the airframe's AGL
    # stand-in, or observed terrain) also bounds local routes above the ground.
    ground = mission._ground if mission is not None and mission.desired_agl is not None else None
    band_margin = float(cfg.spline.get('agl_band_margin', .1)) if 'spline' in cfg else .1
    limits = SimpleNamespace(**dict(cfg.task.flight))
    if spline_mode:
        limits.max_speed = min(limits.max_speed, guidance.speed_limit)
    limits.max_acceleration *= cfg.lower.acceleration_fraction
    limits.max_jerk *= cfg.lower.jerk_fraction
    heading = HeadingReference(env.scene_info['initial_yaw'], cfg.local.heading_max_rate,
                               cfg.local.heading_max_acceleration)
    estimator = ActualAccelerationEstimator(samples=5)
    state = env.state[0, 0, :13].detach().cpu().numpy()
    column = cfg.local.get('takeoff_column')
    if column is not None:
        # The vehicle starts in the air above its vertical take-off column.
        marked = memory.mark_observed_column(state[:3], env.elapsed, float(column.depth), float(column.radius))
        print(f'TAKEOFF_COLUMN voxels={marked}', flush=True)
    command = VelocityTrajectory(state[:3], state[7:10], np.zeros(3), np.zeros(3), limits)
    command_start, generation = env.elapsed, 0
    initial_bearing = float(np.arctan2(goal_spec.target_world[1]-state[1], goal_spec.target_world[0]-state[0]))
    aligned = not cfg.local.align_before_motion
    latest = None
    audit_frames, references, actual_accelerations, active_goals = deque(), [], [], []
    ages, computes, control_times, outputs = [], [], [], []
    counters = Counter()
    tracking_rows = []
    esdf_updates, esdf_times, esdf_peak_voxels = 0, [], 0
    stop_plan, stop_monitor, stop_begin, stop_mode = None, None, None, None
    terminal_reason, failure_reason, reason = None, None, 'interrupted'
    last_status, stopped = None, False
    reference_settled = False
    settled_stop_monitor = None
    settled_stop_plan, settled_stop_begin = None, None
    horizon = 2/cfg.local.hz
    lease_until = -np.inf
    scan_period = 1/cfg.local.hz
    brake_reserve = float(cfg.lower.get('observation_brake_reserve', 0.))
    planning_reserve = float(cfg.lower.get('planning_clearance_reserve', 0.))
    observation_stop_cause(0., cfg.local.max_observation_age, brake_reserve)
    next_scan = env.elapsed
    events_logged = 0
    write_json(output/'guidance.json', asdict(guidance) | dict(points=guidance.points.tolist(),
        corridor=guidance.corridor.tolist() if guidance.corridor is not None else None))

    def prepare(frames, measured, guide, captured):
        nonlocal esdf_updates, esdf_peak_voxels
        for frame in frames:
            memory.add(frame)
        now = float(frames[-1]['timestamp'])
        target, gate_hold = gate_target(guide.goal, measured[:3], guide.target, cfg.local.lookahead)
        guide.target = target
        points, evidence = memory.snapshot(now, measured[:3])
        field = None
        field_cfg = cfg.lower.get('esdf')
        field_seconds = None
        if field_cfg is not None and field_cfg.enabled:
            field_began = time.perf_counter()
            # Rebuild an immutable local EDT snapshot from the SAME rolling
            # online evidence, including expired/deleted hits. No truth mesh.
            field = ObservedESDF(points, evidence, resolution=float(field_cfg.resolution),
                bounds=np.array([measured[:3]-[5., 4., 2.], measured[:3]+[5., 4., 2.]]),
                max_voxels=int(field_cfg.max_voxels))
            field_seconds = time.perf_counter()-field_began
            esdf_times.append(field_seconds)
            esdf_updates += 1
            esdf_peak_voxels = max(esdf_peak_voxels, int(np.prod(field.shape)))
        _, decision, safety = planner.plan(points, evidence, measured[:3], yaw_from_quat(measured[3:7]),
            measured[7:10], np.zeros(3), guide, geometry_only=True, distance_field=field,
            clearance_weight=float(field_cfg.clearance_weight) if field is not None else 0.,
            preferred_clearance=float(field_cfg.preferred_clearance) if field is not None else 1.5,
            planning_reserve=planning_reserve, timestamp=now, gate_hold=gate_hold, ground=ground)
        decision['esdf_update_seconds'] = field_seconds
        planning_safety = safety.with_clearance_reserve(planning_reserve)
        if ground is not None and planner.agl_band is not None:
            # The spline worker (another process) checks height above the ground
            # on this grid: the route band widened by the spline margin, over the
            # ESDF window the spline's controls must stay in.
            xs = measured[0]+np.arange(-5., 5.01, .25)
            ys = measured[1]+np.arange(-4., 4.01, .25)
            grid = np.stack(np.meshgrid(xs, ys, indexing='ij'), -1).reshape(-1, 2)
            low, high = planner._band_limits(np.c_[grid, np.zeros(len(grid))], guide.guidance.points, ground)
            planning_safety.height_band = HeightBand(xs, ys, (low-band_margin).reshape(len(xs), len(ys)),
                                                     (high+band_margin).reshape(len(xs), len(ys)))
        return dict(points=points, evidence=evidence, safety=safety, decision=decision, state=measured,
            planning_safety=planning_safety,
            timestamp=now, captured=captured, goal_id=guide.guidance.segment_id,
            revision=guide.guidance.revision, map_origin=memory.origin.copy(),
            compute_seconds=time.perf_counter()-captured, target=target, gate_hold=gate_hold,
            distance_field=field)

    def record(stream, now, event, **data):
        stream.write(json.dumps(dict(timestamp=float(now), event=event, generation=generation,
            goal_id=goal_spec.goal_id, revision=goal_spec.revision, **data))+'\n')

    def latch_stop(stream, now, mode, cause, actual_a):
        nonlocal command, command_start, generation, stop_plan, stop_monitor, stop_begin, stop_mode
        nonlocal failure_reason, terminal_reason
        nonlocal reference_settled, settled_stop_monitor, settled_stop_plan, settled_stop_begin
        if stop_plan is not None:
            return
        reference_settled = False
        settled_stop_monitor = settled_stop_plan = settled_stop_begin = None
        stop_plan = start_stop(command, max(0., now-command_start), latest['safety'] if latest else None,
            actual_position=state[:3], actual_velocity=state[7:10], actual_acceleration=actual_a,
            flight=cfg.task.flight, reference_flight=limits)
        source_age = (observation_age(now, time.perf_counter(), latest['timestamp'], latest['captured'])
                      if latest else None)
        rest_speed = cfg.lower.get('stale_rest_speed')
        speeds = (float(np.linalg.norm(state[7:10])),
                  float(np.linalg.norm(command.sample(max(0., now-command_start))[1])))
        if source_age is not None and stale_stop_uncertified(source_age, cfg.local.max_observation_age,
                *speeds, rest_speed):
            stop_plan.normal_certified, stop_plan.reason = False, 'STALE_OBSERVATION'
        elif source_age is not None and source_age > cfg.local.max_observation_age:
            record(stream, now, 'stale_observation_at_rest', observation_age_seconds=source_age,
                actual_speed=speeds[0], reference_speed=speeds[1])
        command, command_start = stop_plan.command, now
        if not stop_plan.normal_certified:
            failure_reason = failure_reason or 'uncertified_stop'
            terminal_reason, mode = failure_reason, 'terminal'
        generation += 1
        stop_begin, stop_mode = now, mode
        stop_monitor = MeasuredStopMonitor(dwell=cfg.lower.get('stop_dwell', .3))
        counters['stop_references'] += 1
        counters['uncertified_stop_references'] += int(not stop_plan.normal_certified)
        record(stream, now, 'stop_latched', mode=mode, cause=cause,
            observation_age_seconds=source_age,
            normal_certified=stop_plan.normal_certified, check_reason=stop_plan.reason,
            normal_violation=stop_plan.normal_violation,
            actual_position=state[:3].tolist(), actual_velocity=state[7:10].tolist(),
            actual_acceleration=actual_a.tolist(), nominal_envelope=stop_plan.nominal_envelope.tolist(),
            actual_envelope=stop_plan.actual_envelope.tolist())

    if spline_mode:
        from .stable_spline import StableSplinePlanner
        coordinator = StableSplinePlanner(limits, activation_lead=cfg.spline.activation_lead,
            candidate_max_age=cfg.spline.candidate_max_age, control_dt=env.dt,
            process_worker=cfg.spline.process_worker,
            profile_acceleration=cfg.spline.get('profile_acceleration'))
    else:
        coordinator = nullcontext()
    from .runtime_gc import RestGarbageCollector
    gc_runtime = RestGarbageCollector(defer=cfg.lower.get('defer_gc_while_moving', False),
        defer_full=cfg.lower.get('defer_full_gc_while_moving', False),
        interval=cfg.lower.get('full_gc_interval', 30.), trace=cfg.lower.get('trace_gc', False),
        force_interval=cfg.lower.get('full_gc_force_interval'))
    with gc_runtime, ThreadPoolExecutor(max_workers=1) as pool, coordinator as spline_planner, (output/'decisions.jsonl').open('w') as stream, TraceJournal(output) as journal:
        pipeline = ObservationPipeline(pool, prepare)
        while True:
            clock, now = time.perf_counter(), env.elapsed
            gc_runtime.now = float(now)
            state = env.state[0, 0, :13].detach().cpu().numpy()
            yaw = yaw_from_quat(state[3:7])
            actual_a = estimator.update(now, state[7:10])
            reference_p, reference_v, reference_a = command.sample(max(0., now-command_start))
            tracker.update(state[:3], yaw, now)
            checks = dict(actual_position=state[:3], actual_velocity=state[7:10], actual_acceleration=actual_a,
                horizon=horizon, position_tolerance=cfg.lower.position_tolerance,
                velocity_tolerance=cfg.lower.velocity_tolerance)
            if spline_mode:
                checks['reference_flight'] = limits
            # Collision/nonfinite state cannot be represented as an ordinary controlled stop.
            if env.reason and env.reason not in ('timeout',):
                failure_reason, reason = env.reason, env.reason
                record(stream, now, 'unrecoverable_environment_failure', reason=reason)
                break
            if failure_reason is None and now >= cfg.task.flight.settle_seconds:
                if np.linalg.norm(actual_a) > cfg.task.flight.max_acceleration*1.001:
                    failure_reason = 'actual_acceleration_limit'
                elif np.linalg.norm(state[7:10]) > cfg.task.flight.max_speed*1.001:
                    failure_reason = 'actual_speed_limit'
                elif spline_mode and np.linalg.norm(state[:3]-reference_p) > cfg.lower.position_tolerance:
                    failure_reason = 'tracking_position_limit'
                elif spline_mode and np.linalg.norm(state[7:10]-reference_v) > cfg.lower.velocity_tolerance:
                    failure_reason = 'tracking_velocity_limit'
                elif cfg.lower.get('fault_after_seconds') is not None and now >= cfg.lower.fault_after_seconds:
                    failure_reason = 'injected_stop'
                elif now >= cfg.max_seconds or env.reason == 'timeout':
                    failure_reason = 'timeout'
                if failure_reason:
                    terminal_reason = failure_reason
                    latch_stop(stream, now, 'terminal', failure_reason, actual_a)
            feedback = (mission.update(now, state[:3], state[7:10], yaw, failure_reason) if mission is not None
                        else monitor.update(now, state[:3], state[7:10], yaw, failure_reason))
            status = mission.status if mission is not None else feedback['status']
            if (status in ('COMPLETE', 'FAILED') and terminal_reason is None
                    and (status == 'FAILED' or not gc_runtime.waiting_for_observation)):
                cause = ('mission_reached' if mission is not None else 'goal_reached') if status == 'COMPLETE' else feedback['reason']
                if reference_settled:
                    # A slow policy can finish or fail AFTER the vehicle has
                    # already stopped. Reuse that exact latched reference
                    # and its continuously monitored rest evidence; constructing
                    # another brake would reset the lifecycle and can itself
                    # age an otherwise fresh map during unnecessary work.
                    fresh = latest is not None and observation_age(now, time.perf_counter(),
                        latest['timestamp'], latest['captured']) <= cfg.local.max_observation_age
                    if fresh:
                        checked = execution_check(command, max(0., now-command_start), latest['safety'], **checks)
                        fresh = observation_age(now, time.perf_counter(), latest['timestamp'],
                                                latest['captured']) <= cfg.local.max_observation_age
                        quiet = settled_stop_monitor.update(now, state[:3], state[7:10], actual_a,
                                                            reference_v, reference_a)
                        if checked.accepted and fresh and quiet:
                            terminal_reason = cause
                            failure_reason = cause if status == 'FAILED' else None
                            stop_plan, stop_monitor = settled_stop_plan, settled_stop_monitor
                            stop_begin, stop_mode = settled_stop_begin, 'terminal'
                            record(stream, now, 'terminal_stop_reused', cause=cause,
                                original_command_start=command_start, normal_certified=True,
                                revalidated_source_time=latest['timestamp'],
                                actual_position=state[:3].tolist(), actual_velocity=state[7:10].tolist())
                        elif not checked.accepted:
                            terminal_reason = cause
                            failure_reason = cause if status == 'FAILED' else None
                            latch_stop(stream, now, 'terminal', cause, actual_a)
                    # A stale stationary snapshot is not a new motion permit.
                    # Continue sensing/holding until fresh completion evidence
                    # arrives, or the ordinary mission timeout/fault fires.
                else:
                    terminal_reason = cause
                    failure_reason = terminal_reason if status == 'FAILED' else None
                    latch_stop(stream, now, 'terminal', terminal_reason, actual_a)
            through = goal_spec.completion_mode == 'through' and not getattr(mission, 'requires_handoff_stop', False)
            if not aligned and now >= cfg.task.flight.settle_seconds:
                aligned = abs(angle_error(initial_bearing, yaw)) < .15 and abs(state[12]) < .15
            waiting = mission is not None and mission.status == 'WAITING_NEXT'
            if reference_settled:
                # Keep every completed stop's dwell evidence live, including
                # recovery holds while the current mission is still RUNNING.
                # Disturbance/drift invalidates it and requires a new measured
                # stop before a policy goal may activate.
                reference_settled = settled_stop_monitor.update(now, state[:3], state[7:10],
                    actual_a, reference_v, reference_a)
                if not reference_settled:
                    stopped = False
            if waiting and not through and stop_plan is None and not reference_settled:
                latch_stop(stream, now, 'handoff', getattr(mission, 'handoff_reason', 'goal_reached'), actual_a)
            if (terminal_reason is None and gc_runtime.full_collection_due(now)
                    and stop_plan is None and now >= cfg.task.flight.settle_seconds
                    and (reference_settled or latest is not None and observation_age(
                        now, time.perf_counter(), latest['timestamp'], latest['captured'])
                        <= cfg.local.max_observation_age-brake_reserve)):
                if reference_settled:
                    # Already measured at rest: preserve its original reference,
                    # monitor and command clock rather than construct a new stop.
                    stop_plan, stop_monitor = settled_stop_plan, settled_stop_monitor
                    stop_begin, stop_mode = settled_stop_begin, 'handoff'
                    generation += 1
                else:
                    latch_stop(stream, now, 'maintenance', 'FULL_GC_DUE', actual_a)
                record(stream, now, 'maintenance_start', cause='FULL_GC_DUE',
                    normal_certified=stop_plan.normal_certified,
                    original_command_start=command_start)
            if stop_plan is not None:
                stopped = stop_monitor.update(now, state[:3], state[7:10], actual_a, reference_v, reference_a)
                if stopped:
                    if not gc_runtime.waiting_for_observation:
                        record(stream, now, 'measured_stop', mode=stop_mode, terminal_reason=terminal_reason,
                            stop_seconds=now-stop_begin, actual_position=state[:3].tolist(),
                            actual_velocity=state[7:10].tolist(), actual_acceleration=actual_a.tolist())
                    if terminal_reason is not None:
                        reason = terminal_reason
                        break
                    collected = gc_runtime.collect_at_rest(now, stop_plan.normal_certified)
                    if collected is not None:
                        record(stream, now, 'gc_at_measured_rest', **collected)
                    if not gc_runtime.waiting_for_observation:
                        # Keep the completed reference and its clock while
                        # sensing continues; ordinary recovery is also a hold.
                        reference_settled = True
                        settled_stop_monitor = stop_monitor
                        settled_stop_plan, settled_stop_begin = stop_plan, stop_begin
                    if not gc_runtime.waiting_for_observation:
                        stop_plan, stop_monitor, stop_mode, stop_begin = None, None, None, None
                elif now-stop_begin > cfg.lower.stop_timeout:
                    failure_reason, reason = failure_reason or 'stop_timeout', 'stop_timeout'
                    record(stream, now, 'stop_timeout', initial_reason=terminal_reason,
                           actual_velocity=state[7:10].tolist())
                    break
                if (gc_runtime.waiting_for_observation and terminal_reason is None
                        and time.perf_counter()-gc_runtime.resume_after_wall > cfg.lower.stop_timeout):
                    failure_reason = terminal_reason = 'maintenance_observation_timeout'
                    stop_mode = 'terminal'
                    record(stream, now, 'maintenance_revalidation_failed', reason=failure_reason)
            if (waiting and terminal_reason is None and stop_plan is None
                    and not gc_runtime.waiting_for_observation and not gc_runtime.full_collection_due(now)
                    and (through or stopped or reference_settled)):
                previous_mode = 'stop' if getattr(mission, 'requires_handoff_stop', False) else goal_spec.completion_mode
                # Activation runs on the control thread; its wall time counts
                # toward observation age (run 4eec7e0a went stale at 0.31 s).
                activation_began = time.perf_counter()
                profiler = None
                if cfg.lower.get('profile_activation', False):
                    import cProfile
                    profiler = cProfile.Profile()
                    profiler.enable()
                activated = mission.advance(now, state[:3], state[7:10], yaw,
                    reference_settled=(previous_mode == 'through' or stopped or reference_settled))
                if profiler is not None:
                    profiler.disable()
                advance_seconds = time.perf_counter()-activation_began
                if activated is not None:
                    goal_spec, monitor = activated, mission.monitor
                    reference_settled, stopped = False, False
                    settled_stop_monitor = None
                    settled_stop_plan, settled_stop_begin = None, None
                    guidance_began = time.perf_counter()
                    guidance = replace(tracker.guidance, revision=activated.revision,
                        segment_id=activated.goal_id, points=mission.guidance_points(state[:3]), source_time=now)
                    guidance_seconds = time.perf_counter()-guidance_began
                    tracker.set(guidance, now); tracker.update(state[:3], yaw, now)
                    initial_bearing = float(np.arctan2(activated.target_world[1]-state[1], activated.target_world[0]-state[0]))
                    aligned = previous_mode == 'through' or not cfg.local.align_before_motion
                    through = activated.completion_mode == 'through'
                    env.scene_info['goal'] = list(activated.target_world)
                    generation += 1
                    events_began = time.perf_counter()
                    # Append only new events; the full list is written at the end.
                    # Rewriting ~1 MB here took 50-65 ms late in a mission and every
                    # activation then crossed the observation deadline (run 79f68b73).
                    with (output/'mission_events.jsonl').open('a') as events_file:
                        for event in mission.events[events_logged:]:
                            events_file.write(json.dumps(event, ensure_ascii=False)+'\n')
                    events_logged = len(mission.events)
                    timing = dict(advance=advance_seconds, guidance=guidance_seconds,
                                  events_write=time.perf_counter()-events_began,
                                  total=time.perf_counter()-activation_began)
                    record(stream, now, 'goal_activated', completion_mode=activated.completion_mode,
                        activation_wall_seconds=timing,
                        activation_profile=_profile_top(profiler) if profiler is not None else None)
                    print(f'LOWER_NEXT t={now:.2f} goal={activated.goal_id}', flush=True)
            waiting = mission is not None and mission.status == 'WAITING_NEXT'
            task_running = mission is None or mission.status == 'RUNNING'
            result = pipeline.poll()
            if result is not None:
                latest = result
                if mission is not None and hasattr(mission, 'observe'):
                    # Upper-policy observation is an immutable online snapshot;
                    # its heavy input rendering/model work must be asynchronous.
                    mission.observe(result)
                age = observation_age(now, time.perf_counter(), result['timestamp'], result['captured'])
                ages.append(age); computes.append(result['compute_seconds'])
                if (gc_runtime.waiting_for_observation and terminal_reason is None
                        and stop_plan is not None and stopped):
                    # Hold the same measured-rest reference through collection
                    # and late worker results. Only a genuinely new capture may
                    # grant the next motion lease; numeric age limits are intact.
                    checked = None
                    if result['captured'] > gc_runtime.resume_after_wall and age <= cfg.local.max_observation_age:
                        checked = execution_check(command, max(0., now-command_start), result['safety'], **checks)
                        if not checked.accepted and checked.reason not in ('REFERENCE_UNOBSERVED', 'ACTUAL_UNOBSERVED'):
                            failure_reason = terminal_reason = 'maintenance_revalidation:'+checked.reason
                            stop_mode = 'terminal'
                            record(stream, now, 'maintenance_revalidation_failed', reason=checked.reason,
                                nominal_envelope=checked.nominal_envelope.tolist(),
                                actual_envelope=checked.actual_brake_envelope.tolist())
                    fresh_age = observation_age(now, time.perf_counter(), result['timestamp'], result['captured'])
                    gc_end = gc_runtime.resume_after_wall
                    if terminal_reason is None and gc_runtime.release_after_observation(captured=result['captured'], age=fresh_age,
                            max_age=cfg.local.max_observation_age, certified=bool(checked and checked.accepted),
                            measured_rest=stopped):
                        record(stream, now, 'maintenance_complete', original_command_start=command_start,
                            gc_end_wall=gc_end,
                            post_gc_snapshot_source=result['timestamp'], post_gc_snapshot_captured=result['captured'],
                            observation_age_seconds=fresh_age)
                        reference_settled = True
                        settled_stop_monitor = stop_monitor
                        settled_stop_plan, settled_stop_begin = stop_plan, stop_begin
                        stop_plan, stop_monitor, stop_mode, stop_begin = None, None, None, None
                current = result['goal_id'] == goal_spec.goal_id and result['revision'] == goal_spec.revision
                decision = dict(result['decision'])
                rejection, envelope, certified = 'SUPERSEDED', np.empty((0, 3)), False
                if spline_mode and terminal_reason is None and stop_plan is None and now >= cfg.task.flight.settle_seconds:
                    # Optimization uses an immutable snapshot; execution leases
                    # always depend on this fresh observation and stopping check.
                    rejection = 'STALE'
                    if age <= cfg.local.max_observation_age:
                        checked = execution_check(command, max(0., now-command_start), result['safety'], **checks)
                        rejection, envelope, certified = checked.reason, checked.nominal_envelope, checked.accepted
                        if (certified and planning_reserve and np.linalg.norm(state[7:10]) > .05
                                and not result['safety'].with_clearance_reserve(planning_reserve/2).clear(
                                    checked.actual_brake_envelope)):
                            rejection, certified = 'TRACKING_CLEARANCE_RESERVE', False
                        lease_until = now+horizon if certified else -np.inf
                        near = not through and np.linalg.norm(state[:3]-goal_spec.target_world) <= goal_spec.position_tolerance
                        if (certified and current and aligned and not near and task_running
                                and not gc_runtime.full_collection_due(now)):
                            if spline_planner.submit(command, command_start, now, result, generation,
                                    goal_spec.revision, result['target'], tracker.guidance.speed_limit):
                                counters['spline_requests'] += 1
                    if not certified and not (reference_settled and rejection == 'STALE'):
                        latch_stop(stream, now, 'recovery', rejection, actual_a)
                        if not stop_plan.normal_certified:
                            failure_reason = terminal_reason = 'uncertified_stop'
                            stop_mode = 'terminal'
                elif not spline_mode and current and task_running and terminal_reason is None and stop_plan is None and now >= cfg.task.flight.settle_seconds:
                    if age > cfg.local.deadline:
                        rejection = 'STALE'
                        counters['deadline_misses'] += 1
                    else:
                        near = not through and np.linalg.norm(state[:3]-goal_spec.target_world) <= goal_spec.position_tolerance
                        rejection = 'NO_CANDIDATE'
                        choices = velocity_candidates(decision['raw_velocity'], state[:3], result['target'],
                            tracker.guidance.speed_limit, allow_motion=aligned and not near)
                        for target_velocity in choices:
                            candidate = VelocityTrajectory(state[:3], reference_v, reference_a, target_velocity, limits)
                            rejection, envelope = certificate(result['safety'], candidate, state[:3], horizon)
                            if rejection:
                                continue
                            checked = execution_check(candidate, 0., result['safety'], **checks)
                            rejection = checked.reason
                            if not rejection:
                                command, command_start = candidate, now
                                reference_settled = False
                                settled_stop_monitor = settled_stop_plan = settled_stop_begin = None
                                generation += 1
                                lease_until = now+horizon
                                certified = True
                                counters['accepted_commands'] += 1
                                outputs.append(time.perf_counter())
                                break
                    if not certified:
                        latch_stop(stream, now, 'recovery', rejection, actual_a)
                        if not stop_plan.normal_certified:
                            failure_reason = terminal_reason = 'uncertified_stop'
                            stop_mode = 'terminal'
                record(stream, now, 'observation', source_time=result['timestamp'], pipeline_seconds=age,
                    compute_seconds=result['compute_seconds'], current_goal=current, certified=certified,
                    reason=rejection, decision=decision, actual_position=state[:3].tolist(),
                    actual_velocity=state[7:10].tolist(), actual_acceleration=actual_a.tolist(),
                    reference_velocity=reference_v.tolist(), reference_acceleration=reference_a.tolist(),
                    envelope=envelope.tolist())
            if spline_mode:
                candidate, event = spline_planner.poll(now, time.perf_counter(), generation,
                    goal_spec.revision, command, command_start)
                if event is not None:
                    record(stream, now, 'spline_candidate', details=event)
                if candidate is not None:
                    fresh = latest is not None and observation_age(now, time.perf_counter(),
                        latest['timestamp'], latest['captured']) <= cfg.local.max_observation_age
                    if (terminal_reason is None and stop_plan is None and not gc_runtime.waiting_for_observation
                            and not gc_runtime.full_collection_due(now) and fresh and task_running):
                        checked = execution_check(candidate.path, 0., latest['safety'], **checks)
                        reserved = latest['planning_safety'].clear(checked.nominal_envelope) if checked.accepted else False
                        if checked.accepted and reserved:
                            command, command_start = candidate.path, candidate.request.switch_time
                            reference_settled = False
                            settled_stop_monitor = settled_stop_plan = settled_stop_begin = None
                            generation += 1
                            lease_until = now+horizon
                            counters['spline_activations'] += 1
                            counters['accepted_commands'] += 1
                            record(stream, now, 'spline_activated', switch_time=command_start,
                                start_pva=candidate.request.start_pva.tolist(), spline=command.as_dict())
                        else:
                            record(stream, now, 'spline_activation_rejected',
                                reason=checked.reason or 'PLANNING_CLEARANCE_RESERVE')
                    else:
                        record(stream, now, 'spline_activation_rejected', reason='STOPPING_OR_STALE_MAP')
            if terminal_reason is None and stop_plan is None and now >= cfg.task.flight.settle_seconds:
                moving = np.linalg.norm(command.sample(max(0., now-command_start))[1]) > .01
                deadline = ('STALE' if latest is None else observation_stop_cause(
                    observation_age(now, time.perf_counter(), latest['timestamp'], latest['captured']),
                    cfg.local.max_observation_age, brake_reserve))
                if moving and (deadline or now > lease_until+1e-8):
                    latch_stop(stream, now, 'recovery', deadline or 'LEASE_EXPIRED', actual_a)
                    if not stop_plan.normal_certified:
                        failure_reason = terminal_reason = 'uncertified_stop'
                        stop_mode = 'terminal'
            if now >= next_scan-1e-7:
                captured = time.perf_counter(); frame = env.scan()
                if mission is not None and hasattr(mission, 'observe_frame'):
                    mission.observe_frame(frame)
                audit_frames.append(frame)
                while audit_frames and float(audit_frames[0]['timestamp']) < now-cfg.history_seconds-1.:
                    audit_frames.popleft()
                next_scan = now+scan_period
                request_tracker = copy(tracker)
                request_tracker.goal = goal_spec
                pipeline.push(frame, measured=state.copy(), guide=request_tracker, captured=captured)
            p, v, a = command.sample(max(0., now-command_start))
            target_yaw = (goal_spec.arrival_yaw if not through and np.linalg.norm(state[:3]-goal_spec.target_world) < 1.
                and goal_spec.arrival_yaw is not None else np.arctan2(v[1], v[0]) if np.linalg.norm(v[:2]) > .08 else heading.yaw)
            if not aligned and now >= cfg.task.flight.settle_seconds and stop_plan is None:
                target_yaw = initial_bearing
            if terminal_reason is not None:
                target_yaw = heading.yaw
            reference_yaw, rate, yaw_a = heading.step(target_yaw, env.dt)
            env.set_heading_feedforward(rate, yaw_a)
            references.append(p); actual_accelerations.append(actual_a); active_goals.append(goal_spec.goal_id)
            if spline_mode:
                tracking_rows.append([now, float(np.linalg.norm(state[:3]-p)),
                    float(np.linalg.norm(state[7:10]-v)), float(np.linalg.norm(actual_a))])
            row = step_velocity(env, v, a, reference_yaw, p)
            traces.append(row)
            journal.append(row)
            control_times.append(time.perf_counter()-clock)
            current_status = (goal_spec.goal_id, stop_mode, terminal_reason)
            if current_status != last_status:
                print(f'LOWER t={now:.2f} goal={goal_spec.goal_id} stop={stop_mode} reason={terminal_reason or "RUNNING"}', flush=True)
                last_status = current_status
            if cfg.realtime:
                # A slow step must not create a later burst of accelerated
                # simulation. Map workers and sensors use wall time; catching
                # up a global deadline can age their snapshots in simulation
                # before the next wall-clock result can arrive.
                time.sleep(max(0., clock+env.dt-time.perf_counter()))
    if latest is not None:
        np.savez_compressed(output/'last_observation.npz', points=latest['points'], evidence=latest['evidence'],
            position=latest['state'][:3], yaw=yaw_from_quat(latest['state'][3:7]), timestamp=latest['timestamp'],
            map_origin=latest['map_origin'])
        if latest['distance_field'] is not None:
            latest['distance_field'].save(output/'last_esdf.npz')
            write_json(output/'esdf.json', dict(schema=latest['distance_field'].schema,
                source='same online rolling-map hits and ray evidence as last_observation.npz',
                timestamp=float(latest['timestamp']), update_method='bounded local EDT rebuild, not FIESTA incremental BFS',
                role=('A* clearance cost and B-spline optimization; independent known/body/braking checks'
                      if spline_mode else 'A* clearance cost; independent known/body/braking checks remain authoritative'),
                distance_semantics='conservative distance to observed hits; not distance to unobserved surfaces or reconstructed solid interiors',
                gradient=('analytic trilinear; used in B-spline obstacle cost' if spline_mode
                          else 'analytic trilinear; available for subsequent trajectory optimization'),
                updates=esdf_updates, peak_voxels=esdf_peak_voxels,
                update_p95_seconds=float(np.percentile(esdf_times, 95))))
    if audit_frames:
        folder = output/'observation_window'; folder.mkdir()
        for i, frame in enumerate(audit_frames):
            np.savez_compressed(folder/f'{i:04d}.npz', **frame)
        write_json(folder/'manifest.json', dict(frames=len(audit_frames),
            first_timestamp=float(audit_frames[0]['timestamp']), last_timestamp=float(audit_frames[-1]['timestamp']),
            ray_traversal=memory.ray_traversal, rolling=True, window_size=list(cfg.lower.map_window),
            scope='last history_seconds + 1 second before stopping; not the full flight'))
    np.save(output/'reference_positions.npy', np.asarray(references).reshape(-1, 3))
    np.save(output/'control_wall_seconds.npy', np.asarray(control_times))
    write_json(output/'gc_events.json', dict(deferred=gc_runtime.defer, deferred_full=gc_runtime.defer_full,
        original_threshold=gc_runtime.original_threshold, full_gc_interval=gc_runtime.interval,
        full_gc_force_interval=gc_runtime.force_interval, events=gc_runtime.events,
        scope='cyclic GC scheduling only; reference counting and all execution age/stop checks unchanged'))
    np.save(output/'actual_accelerations.npy', np.asarray(actual_accelerations).reshape(-1, 3))
    write_json(output/'active_goals.json', active_goals)
    if spline_mode:
        np.save(output/'tracking_errors.npy', np.asarray(tracking_rows).reshape(-1, 4))
    summary = dict(reason=reason, failure_reason=failure_reason, physical_stop_verified=bool(stopped),
        realtime_pacing='per_step_no_catchup' if cfg.realtime else 'unpaced',
        map_recenters=memory.recenter_count, map_shape=list(memory.seen.shape),
        map_timestamp_bytes=memory.seen.nbytes+memory.occupied.nbytes,
        esdf_updates=esdf_updates, esdf_peak_voxels=esdf_peak_voxels,
        esdf_update_p95_seconds=float(np.percentile(esdf_times, 95)) if esdf_times else None,
        estimator='five-step actual world-velocity differences; not reference acceleration', **counters,
        goal_distance=float(np.linalg.norm(state[:3]-goal_spec.target_world)),
        pipeline_p95_seconds=float(np.percentile(ages, 95)) if ages else None,
        compute_p95_seconds=float(np.percentile(computes, 95)) if computes else None,
        control_p95_seconds=float(np.percentile(control_times, 95)) if control_times else None,
        control_overruns=sum(t > env.dt for t in control_times),
        actual_max_acceleration_mps2=float(np.linalg.norm(actual_accelerations, axis=1).max()) if actual_accelerations else None)
    summary['subgoal'] = monitor.update(env.elapsed, state[:3], state[7:10], yaw_from_quat(state[3:7]), failure_reason)
    if spline_mode and tracking_rows:
        error = np.asarray(tracking_rows)
        summary['tracking'] = dict(position_mean_m=float(error[:, 1].mean()),
            position_p95_m=float(np.percentile(error[:, 1], 95)), position_max_m=float(error[:, 1].max()),
            velocity_max_mps=float(error[:, 2].max()),
            position_limit_exceedances=int((error[:, 1] > cfg.lower.position_tolerance).sum()),
            velocity_limit_exceedances=int((error[:, 2] > cfg.lower.velocity_tolerance).sum()),
            semantics='measured and analytic p/v reference compared at the same pre-control simulation time')
    if mission is not None:
        if mission.status not in ('COMPLETE', 'FAILED'):
            mission.update(env.elapsed, state[:3], state[7:10], yaw_from_quat(state[3:7]), failure_reason or reason)
        summary['mission'] = mission.feedback()
        write_json(output/'mission_events.json', mission.events)
        write_json(output/'goal_results.json', mission.results)
    return summary
