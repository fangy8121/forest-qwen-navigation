"""主线程 Isaac 步进、采集和异步 Qwen 闭环。"""
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .geometry import BOUNDS, Z_BOUNDS, VERTICAL_RADIUS, yaw_from_quat, INFLATION, world_to_local, corridor_polygon
from .trajectory import Heading, QuinticPath
from .planning import RollingPlanner, smooth_path
from .observation import History, render_observation
from .execution import ReferencePlayer, ContinuousReferencePlayer
from .scan_writer import ScanWriter


def new_run(root):
    path = Path(root).expanduser().resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def local_goal(world_goal, position, yaw, horizon=6.):
    delta = world_to_local(np.asarray([world_goal]), position, yaw)[0]
    scale = min(1., horizon/max(np.linalg.norm(delta), 1e-9))
    # 保持目标方向，截到图像内；不能把图外目标隐去或改变为隐藏中心线。
    for value, lower, upper in zip(delta, (*BOUNDS[::2], Z_BOUNDS[0]), (*BOUNDS[1::2], Z_BOUNDS[1])):
        if value > 0:
            scale = min(scale, (upper-.2)/value)
        elif value < 0:
            scale = min(scale, (lower+.2)/value)
    delta *= scale
    return delta


def run_flight(cfg, closed_loop=False):
    import torch
    from tensordict import TensorDict
    from omegaconf import OmegaConf
    from omni_drones import init_simulation_app
    output = new_run(cfg.output)
    write_json(output/'config.json', OmegaConf.to_container(cfg, resolve=True))
    print(f'FOREST_OUTPUT={output}', flush=True)
    # 模型目录/依赖错误在启动物理前明确失败；不自动下载。
    model = None
    if closed_loop:
        from .qwen import QwenPlanner
        if cfg.get('qwen_python'):
            from .model_process import ModelProcess
            model = ModelProcess(cfg.qwen_python, cfg, output)
        else:
            model = QwenPlanner(cfg.model, cfg.adapter, cfg.qwen_device, cfg.max_input_tokens, cfg.max_new_tokens)
        write_json(output/'model.json', model.metadata)
    app = init_simulation_app(cfg)
    env = None
    reason = 'initialization_failure'
    traces = []
    planner = None
    wall_begin = time.perf_counter()
    control_overruns = 0
    latest_frame = None
    history = None
    actual_path = []
    execution_trace = []
    player = None
    writer = ScanWriter()
    timings = {key: [] for key in ('scan', 'write_queue', 'display', 'physics', 'render', 'step')}
    try:
        from omni_drones.envs.isaac_env import IsaacEnv
        env = IsaacEnv.REGISTRY['ForestSurvey'](cfg, cfg.headless)
        env.reset()
        # Separate physics and viewport clocks. render() pumps UI without
        # integrating physics, so Pause never consumes reference time.
        env.enable_render(False)
        render_every = max(1, round(1/(30*env.dt)))
        (output/'scans').mkdir()
        np.savez_compressed(output/'scene_mesh.npz', vertices=env.mesh_vertices, faces=env.mesh_faces)
        write_json(output/'scene_truth.json', env.scene_info)
        write_json(output/'vehicle.json', dict(dimensions=env.dimensions, authored=env.original_mass,
            physx=env.physics_mass, controller_parameters=env.drone.params,
            rotor_world_positions=env.drone.rotors_view.get_world_poses()[0][0].detach().cpu().numpy().tolist(),
            lidar_source=env.lidar.source))
        import importlib.metadata
        # Keep reproducibility metadata together; no standalone preview/version
        # files are needed in a raw collection directory.
        vehicle_path = output/'vehicle.json'
        vehicle_metadata = json.loads(vehicle_path.read_text())
        vehicle_metadata['software_versions'] = {name: importlib.metadata.version(name)
                   for name in ['isaacsim','isaaclab','torch','numpy','transformers']}
        write_json(vehicle_path, vehicle_metadata)
        from .geometry import center_clearance
        history = History(cfg.history_seconds, center_clearance(cfg.task.flight),
                          cfg.task.flight.get('ray_traversal', 'sampled_v1'))
        corridor_bounds=env.scene_info.get('validation',{}).get('route_bounds')
        start = np.array(env.scene_info['reference'][0], dtype=float)
        heading = Heading(env.scene_info['initial_yaw'], cfg.task.flight.max_yaw_rate,
                          cfg.task.flight.max_yaw_acceleration)
        if closed_loop:
            planner = RollingPlanner(cfg, model, output, start)
        elif 'map_path' in env.scene_info:
            parameters = env.scene_info['flight_parameters']
            for key in ('max_speed', 'max_acceleration', 'max_jerk', 'max_yaw_rate', 'max_yaw_acceleration', 'height'):
                if cfg.task.flight[key] != parameters[key]:
                    raise ValueError(f'offline trajectory {key} differs; regenerate maps with the new flight config')
            route = QuinticPath.load(Path(env.scene_info['map_path']) / env.scene_info['trajectory_file'])
        else:
            from .mesh import horizontal_section, projected_clearance
            section = horizontal_section(env.mesh_vertices, env.mesh_faces,
                cfg.task.flight.height-VERTICAL_RADIUS, cfg.task.flight.height+VERTICAL_RADIUS)
            route = smooth_path(np.asarray(env.scene_info['reference']), cfg.task.flight,
                lambda p: projected_clearance(p[:, :2], section) > INFLATION+.05)
        if not closed_loop:
            route.save(output/'reference_trajectory.npz')
            display_reference = route.remaining(0., start)
            if env.scene_info.get('execution_segments'):
                player = ReferencePlayer(route, env.scene_info['execution_segments'],
                    env.scene_info['reference_phases'], env.scene_info['initial_yaw'], cfg.task.flight.settle_seconds)
            elif env.scene_info.get('execution_strategy') == 'continuous_heading_v1':
                player = ContinuousReferencePlayer(route, env.scene_info['reference_phases'],
                    env.scene_info['initial_yaw'], cfg.task.flight.settle_seconds)
        scan_index = 0
        wall_begin = time.perf_counter()
        reason = 'timeout'
        for step in range(int(cfg.max_seconds/env.dt)):
            while app.is_running() and not env.sim.is_stopped() and not env.sim.is_playing():
                env.sim.render()
                time.sleep(.01)
            if not app.is_running() or env.sim.is_stopped():
                reason = 'user_stopped'; break
            wall_start = time.perf_counter()
            t = env.elapsed
            state = env.state[0,0].detach().cpu().numpy()
            if step % env.scan_steps == 0:
                captured_wall = time.perf_counter()
                frame = env.scan()
                # 任务目标单独随观测保存，教师不需要从隐藏场景路线推导目标。
                frame['task_goal'] = np.asarray(env.scene_info['goal'])
                if player is not None:
                    frame.update(player.metadata())
                latest_frame = frame
                timings['scan'].append(time.perf_counter()-captured_wall)
                stamp = time.perf_counter()
                writer.submit(output/'scans'/f'{scan_index:06d}.npz', frame)
                timings['write_queue'].append(time.perf_counter()-stamp)
                history.add(frame); scan_index += 1
                displayed = (display_reference if not closed_loop else
                             planner.path.remaining(max(0., t-planner.path_start), state[:3])
                             if planner.path is not None else [state[:3]])
                stamp = time.perf_counter()
                env.update_display(frame, history, displayed, actual_path+[state[:3]])
                timings['display'].append(time.perf_counter()-stamp)
                if closed_loop:
                    now_yaw = yaw_from_quat(state[3:7])
                    full_snapshot = planner.needs_snapshot(t)
                    points, evidence = history.snapshot(t, state[:3], now_yaw,
                                                        include_evidence=full_snapshot)
                    if not full_snapshot:
                        evidence = None
                    _, reference_v, reference_a = planner.reference(t)
                    planner.observe(dict(velocity=world_to_local([reference_v], np.zeros(3), now_yaw)[0],
                        acceleration=world_to_local([reference_a], np.zeros(3), now_yaw)[0], points=points, evidence=evidence, position=state[:3].copy(),
                        yaw=now_yaw, timestamp=t, captured_wall=captured_wall,
                        goal_horizon=float(cfg.planning.goal_horizon),
                        corridor=corridor_polygon(corridor_bounds,state[:3],now_yaw) if corridor_bounds is not None else None,
                        goal=local_goal(env.scene_info['goal'], state[:3], now_yaw, cfg.planning.goal_horizon)))
            if not closed_loop:
                if player is not None:
                    p, v, a, target_yaw = player.reference()
                    if player.failure or player.complete:
                        reason = player.failure or 'reference_complete'; break
                else:
                    elapsed = t-cfg.task.flight.settle_seconds
                    p, v, a = route.sample(elapsed)
                    target_yaw = heading.target(route, max(0., elapsed))
                    if t > route.duration+cfg.task.flight.settle_seconds+cfg.history_seconds:
                        reason = 'reference_complete'; break
            else:
                if planner.terminal is None and np.linalg.norm(state[:3]-np.array(env.scene_info['goal'])) < .4:
                    planner.stop(t, 'goal_reached', terminal=True)
                p, v, a = planner.tick(t, state, step % env.scan_steps == 0)
                target_yaw = (heading.target(planner.path, t-planner.path_start)
                              if planner.path is not None and not planner.braking else heading.yaw)
                stopped = np.linalg.norm(state[7:10]) < cfg.task.flight.settle_speed
                if planner.terminal and planner.path is None and stopped:
                    reason = planner.terminal; break
            yaw = target_yaw if player is not None else heading.step(target_yaw, env.dt)
            if player is not None and env.scene_info.get('heading_feedforward', False):
                env.set_heading_feedforward(player.yaw_rate, player.yaw_acceleration)
            ref = np.r_[p,v,a,yaw]
            td = TensorDict({'agents': {'action': torch.tensor(ref, dtype=torch.float32, device=env.device).reshape(1,1,10)}}, [1])
            stamp = time.perf_counter()
            env.step(td)
            timings['physics'].append(time.perf_counter()-stamp)
            if env.elapsed <= t:
                continue
            actual = env.state[0,0].detach().cpu().numpy()
            if player is not None:
                execution_trace.append(dict(simulated_time=env.elapsed, yaw_rate=player.yaw_rate,
                    yaw_acceleration=player.yaw_acceleration, **player.metadata()))
                player.advance(env.elapsed-t, actual, env.elapsed)
            actual_path.append(actual[:3].copy())
            traces.append(np.r_[env.elapsed, ref, actual[:13], env.motor_commands.reshape(-1)])
            if env.reason:
                reason = env.reason; break
            if step % render_every == 0:
                stamp = time.perf_counter()
                env.update_camera()
                env.sim.render()
                timings['render'].append(time.perf_counter()-stamp)
            spent = time.perf_counter()-wall_start
            control_overruns += int(spent > env.dt)
            timings['step'].append(spent)
            if closed_loop:
                # 单步过慢时如实记录，不用仿真时间冒充实际推理更新频率。
                time.sleep(max(0., env.dt-spent))
    finally:
        wall_seconds = time.perf_counter()-wall_begin
        planning_metrics = planner.metrics() if planner is not None else None
        if sys.exc_info()[0] is not None:
            reason = 'exception:'+sys.exc_info()[0].__name__
            import traceback
            traceback.print_exc()
        np.save(output/'trajectory.npy', np.asarray(traces))
        try:
            writer.close()
        except Exception as exc:
            reason = 'scan_write_failure'
            write_json(output/'scan_write_error.json', dict(error=repr(exc)))
        if player is not None:
            write_json(output/'execution.json', dict(events=player.events, trace=execution_trace,
                reference_time=player.time, complete=player.complete, failure=player.failure))
        if cfg.get('save_last_observation', False) and env is not None and latest_frame is not None and history is not None:
            state = env.state[0,0].detach().cpu().numpy()
            yaw = yaw_from_quat(state[3:7])
            points, evidence = history.snapshot(env.elapsed, state[:3], yaw)
            goal = local_goal(env.scene_info['goal'], state[:3], yaw, cfg.planning.goal_horizon)
            # BEV 结束预览与新输入使用相同的三层表示；斜视仍使用原渲染器。
            bounds=env.scene_info.get('validation',{}).get('route_bounds')
            corridor=corridor_polygon(bounds,state[:3],yaw) if bounds is not None else None
            if cfg.view == 'bev':
                if cfg.task.flight.get('input_representation') == 'volume_atlas_v1':
                    from .volume_bev import grid_states, render_atlas
                    final_obs=dict(points=points,evidence=evidence,corridor=corridor,
                                   flight_json=json.dumps(OmegaConf.to_container(cfg.task.flight)))
                    render_atlas(grid_states(final_obs), output/'last_observation_bev.png')
                else:
                    from .waypoint_input_xyz import render_bev
                    render_bev(points, evidence, goal, output/'last_observation_bev.png',
                               corridor, cfg.task.flight.max_acceleration, center_clearance(cfg.task.flight))
            else:
                render_observation(points, evidence, goal, output/f'last_observation_{cfg.view}.png', cfg.view,
                                   max_acceleration=cfg.task.flight.max_acceleration, corridor=corridor)
        summary = dict(reason=reason, steps=len(traces), wall_seconds=wall_seconds,
            timing_seconds={key: dict(mean=float(np.mean(values)), p95=float(np.percentile(values,95)),
                maximum=float(np.max(values))) for key, values in timings.items() if values},
            real_time_factor=(env.elapsed/wall_seconds if env else 0.), control_overruns=control_overruns,
            simulated_seconds=env.elapsed if env else 0.,
            trajectory_columns='t, reference_xyz(3), reference_vel(3), reference_acc(3), yaw, actual_xyz(3), quaternion_wxyz(4), velocity_world(6), motor_commands(4)')
        if traces:
            array = np.asarray(traces)
            errors = np.linalg.norm(array[:,1:4]-array[:,11:14],axis=1)
            summary.update(tracking_error_max=float(errors.max()), tracking_error_mean=float(errors.mean()),
                           scans=len(list((output/'scans').glob('*.npz'))))
        if planning_metrics is not None:
            summary['planning'] = planning_metrics
        write_json(output/'result.json', summary)
        # SimulationApp 的 fast shutdown 可能直接结束进程，因此结果必须在 close 前输出。
        print(f'FOREST_RESULT={reason} {output}', flush=True)
        if sys.exc_info()[0] is not None:
            import omni.kit.app
            omni.kit.app.get_app().post_quit(1)
        if planner is not None:
            planner.close()
        if model is not None and hasattr(model, 'close'):
            model.close()
        if env is not None:
            env.close()
        app.close()
    return output
