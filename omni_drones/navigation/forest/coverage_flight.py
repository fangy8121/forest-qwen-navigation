"""Physical integration and separate, post-flight evaluation for coverage."""
from pathlib import Path
from contextlib import ExitStack
import json
import shutil
import time
import numpy as np
from .flight import new_run, write_json
from .coverage_mission import CoverageMission
from .online_structure import OnlineStructure
from .guidance import Guidance
from .qwen_subgoal import SubgoalModelProcess
from .runs import file_hash


class RuleWorker:
    """Explicit rule baseline, never reported as Qwen output."""
    def __init__(self):
        self.reply = None

    @property
    def busy(self):
        return self.reply is not None

    def submit(self, image=None, prompt=None, *, request_id=None, response_schema='decision', letter_scores=False):
        if self.busy:
            return False
        if response_schema == 'coverage_command':
            data = json.loads(prompt)
            text = data['command']
            intent = ('REVISIT_CORRIDOR' if '通道' in text and ('回' in text or '返回' in text)
                else 'RETURN_HOME' if '返航' in text or '返回起点' in text and '巡检整个' not in text
                else 'RESUME' if text.strip() in ('继续', '继续巡检') else 'COVER')
            value = dict(intent=intent, corridor_id=data['last_departed_corridor'] if intent == 'REVISIT_CORRIDOR' else None,
                resume_after=bool(intent == 'REVISIT_CORRIDOR' and '继续' in text))
        else:
            data = json.loads(prompt[prompt.index('{'):])
            c = data['candidates'][0]
            value = dict(candidate_id=c['candidate_id'], behavior={
                'INSPECT':'FOLLOW', 'OBSERVE':'FOLLOW', 'BOOTSTRAP':'FOLLOW', 'TRANSIT':'CHANGE_AISLE',
                'RETURN':'RETURN', 'REJOIN':'REJOIN', 'STOP':'REQUEST_STOP'}[c['role']])
        self.reply = dict(request_id=request_id, **{response_schema:value}, raw=json.dumps(value),
            source='deterministic rule baseline, not a model', latency_seconds=0.)
        return True

    def poll(self):
        result, self.reply = self.reply, None
        return result

    def close(self):
        pass


class TeacherWorker(RuleWorker):
    """Decision-bench teacher in place of Qwen on the point_choice candidates.

    Same candidates, images and mission flow as ``policy: qwen``; only the
    letter comes from ``decision_bench.teacher``. It reads the decision's saved
    ``input.json`` (online state only) and separates system ceiling from model
    choice quality. A whole-forest command binds as Qwen parses it (COVER,
    S or free order, every aisle); other commands use ``RuleWorker`` keywords.
    """
    def submit(self, image=None, prompt=None, *, request_id=None, response_schema='decision', letter_scores=False):
        if self.busy:
            return False
        if response_schema == 'coverage_command':
            text = json.loads(prompt)['command']
            if '巡检整个' not in text and '巡检全部' not in text:
                return super().submit(image, prompt, request_id=request_id, response_schema=response_schema)
            # RuleWorker reads "返回起点" + "通道" in the S instruction as a revisit.
            value = dict(intent='COVER', target=None, order='free' if '任意顺序' in text else 'S',
                corridors=None, resume_after=False, corridor_id=None)
            self.reply = dict(request_id=request_id, coverage_command=value, raw=json.dumps(value),
                source='teacher command binding, not a model', latency_seconds=0.)
            return True
        if response_schema != 'choice':
            return super().submit(image, prompt, request_id=request_id, response_schema=response_schema)
        from .decision_bench import target_aisle, teacher
        folder = Path(image[0] if isinstance(image, (list, tuple)) else image).parent
        payload = json.loads((folder/'input.json').read_text())
        state, candidates, labels = payload['state'], payload['candidates'], payload['labels']
        public = {p['candidate_id']: p for p in state.get('candidates', [])}
        acceptable, category = teacher(state, candidates, public)
        source = 'teacher'
        if not acceptable:
            # Undefined for the teacher: the pruned S reference, else observe
            # (no aisle seen yet at the start), else wait.
            acceptable = [cid for cid in payload.get('rule_reference') or [] if cid in labels]
            source = 'rule_reference' if acceptable else 'observe'
        if not acceptable:
            looks = [c['candidate_id'] for c in candidates if c['role'] in ('BOOTSTRAP', 'OBSERVE')]
            if looks:
                # The farthest viewpoint changes the view most.
                acceptable = [max(looks, key=lambda cid: ((public.get(cid) or {}).get('total_route_distance_m') or 0.,
                                                          -ord(labels[cid])))]
        if not acceptable:
            source = 'hold'
            acceptable = [c['candidate_id'] for c in candidates if c['role'] in ('HOLD', 'STOP')]
        # Several acceptable: most new coverage, then the shortest route.
        chosen = min(acceptable, key=lambda cid: (-((public.get(cid) or {}).get('new_coverage_m') or 0.),
            (public.get(cid) or {}).get('total_route_distance_m') or 0., labels[cid]))
        topology = {t['corridor_id']: t for t in state.get('topology', [])}
        candidate = next(c for c in candidates if c['candidate_id'] == chosen)
        value = dict(next_aisle=target_aisle(candidate, topology), choice=labels[chosen])
        self.reply = dict(request_id=request_id, choice=value, raw=json.dumps(value),
            teacher=dict(category=category, source=source, acceptable=acceptable, candidate_id=chosen),
            source='decision_bench teacher, not a model', latency_seconds=0.)
        return True


class CommandPolicyWorker:
    """Real Qwen command semantics + explicit rule coverage/anchor baseline.

    Routine geometric anchors do not claim a model decision. This mode tests
    language/interrupt integration without asking a CPU VLM about every small
    deterministic motion. ``qwen`` remains the separate full candidate mode.
    """
    def __init__(self, model):
        self.model, self.rules = model, RuleWorker()

    @property
    def busy(self):
        return self.model.busy or self.rules.busy

    def submit(self, *args, response_schema='decision', **kwargs):
        if self.busy:
            return False
        worker = self.rules if response_schema == 'decision' else self.model
        return worker.submit(*args, response_schema=response_schema, **kwargs)

    def poll(self):
        return self.rules.poll() if self.rules.busy else self.model.poll()

    def close(self):
        self.rules.close()
        self.model.close()


def evaluate_coverage(actual, final_velocity, home, physical_stop_verified, mission,
                      truth_margin_violation, evaluation=None):
    """Hidden geometry is accepted ONLY here, after the physical flight."""
    coverage = []
    if evaluation is not None:
        for c in evaluation['corridors']:
            samples = np.arange(*c['x_range'], .1)
            mask = abs(actual[:, 1]-c['y']) <= .8
            xs = actual[mask, 0]
            seen = np.min(abs(samples[:, None]-xs[None, :]), axis=1) <= .2 if len(xs) else np.zeros(len(samples), bool)
            coverage.append(dict(truth_corridor_id=c['id'], actual_route_coverage_fraction=float(seen.mean())))
    complete = bool(coverage and all(c['actual_route_coverage_fraction'] >= .98 for c in coverage))
    final_home = bool(len(actual) and np.linalg.norm(actual[-1]-home) <= .3)
    stopped = bool(physical_stop_verified and np.linalg.norm(final_velocity) <= .05)
    authorized_end = bool(getattr(mission, 'status', None) == 'COMPLETE'
        and getattr(mission, 'stop_requested', False))
    return dict(truth_route_coverage=coverage, actual_full_route_coverage=complete,
        final_home=final_home, final_measured_stop=stopped,
        truth_margin_clear=truth_margin_violation is False,
        declared_coverage_result=mission.coverage_result,
        authorized_task_termination=authorized_end,
        false_completion=bool(mission.coverage_result == 'COMPLETE_UNDER_ASSUMPTIONS' and not complete),
        full_coverage_passed=bool(complete and final_home and stopped and authorized_end
            and truth_margin_violation is False
            and mission.coverage_result == 'COMPLETE_UNDER_ASSUMPTIONS'),
        metric_scope='centerline route traversal in structural diagnostic; not tree-surface or disease inspection')


def run_coverage(cfg):
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    if cfg.executor != 'stable_spline' or cfg.task.control_mode != 'trajectory':
        raise ValueError('coverage requires the existing stable spline executor')
    output = new_run(cfg.output)
    print(f'COVERAGE_OUTPUT={output}', flush=True)
    write_json(output/'config.json', OmegaConf.to_container(cfg, resolve=True))
    root = Path(__file__).resolve().parents[3]
    paths = list((root/'omni_drones/navigation/forest').glob('*.py'))
    paths += [root/'scripts/forest_subgoal_model_worker.py', root/'scripts/fly_forest_coverage.py']
    paths += list((root/'cfg').glob('forest*.yaml'))
    paths += [root/name for name in ('cfg/task/ForestSurvey.yaml',
        'omni_drones/envs/single/forest_survey.py', 'omni_drones/controllers/lee_position_controller.py')]
    for path in paths:
        destination = output/'source'/path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    source_hashes = {str(p.relative_to(root)):file_hash(p) for p in paths}
    write_json(output/'source_hashes.json', source_hashes)
    native = cfg.coverage.get('scene_kind', 'diagnostic') == 'native'
    if cfg.coverage.get('scene_kind', 'diagnostic') not in ('native', 'diagnostic'):
        raise ValueError('coverage.scene_kind must be native or diagnostic')
    map_folder = Path(cfg.task.scene.map_dir)/cfg.task.scene.map_id
    map_files = ('map.json', 'scene_mesh.npz', 'scene.usd', 'terrain.npz')
    map_hashes = {name:file_hash(map_folder/name) for name in map_files if (map_folder/name).is_file()}
    write_json(output/'map_source.json', dict(path=str(map_folder.resolve()), sha256=map_hashes))
    terrain_source = cfg.coverage.get('terrain_source', 'observed')
    if terrain_source not in ('observed', 'truth'):
        raise ValueError('coverage.terrain_source must be observed or truth')
    terrain_truth = None
    if terrain_source == 'truth':
        with np.load(map_folder/'terrain.npz') as grid:
            terrain_truth = dict(xs=grid['xs'], ys=grid['ys'], heights=grid['heights'])
    agl = cfg.coverage.get('desired_agl') is not None
    protocol = dict(name='native_qwen_s_coverage_v2' if native else 'online_coverage_structural_pilot_v1',
        policy=cfg.coverage.policy,
        height_mode=('TRUTH_TERRAIN_AGL' if terrain_truth is not None else 'OBSERVED_TERRAIN_AGL') if agl else 'WORLD_Z',
        runtime_inputs='instruction, initial pose, declared work-area XY boundary, safety constraints, online LiDAR, measured state'
            +(', known terrain heights (sim stand-in for an airframe AGL measurement)' if terrain_truth is not None else ''),
        no_runtime_inputs='true tree rows/count/endpoints, predetermined site catalog, teacher route, evaluation predicates',
        reset_policy='one reset before sensing; no in-flight reset',
        boundary_scope='conditional sampled absence under declared detector/gap assumptions; unresolved allowed',
        limitations='ideal simulated localization and ray sensor; route coverage, not tree-surface or disease inspection')
    write_json(output/'protocol.json', protocol)
    home = None
    worker = app = env = mission = structure = viewer = None
    traces = []
    result = dict(reason='initialization_failure', passed=False)
    began = time.perf_counter()
    try:
        if cfg.coverage.policy not in ('rule', 'qwen', 'qwen_commands', 'teacher'):
            raise ValueError('policy must be rule, qwen, qwen_commands or teacher')
        if cfg.coverage.policy == 'rule':
            worker = RuleWorker()
        elif cfg.coverage.policy == 'teacher':
            worker = TeacherWorker()
        else:
            worker = SubgoalModelProcess(cfg.qwen.python, cfg.qwen.model, output,
                device=cfg.qwen.device, max_input_tokens=cfg.qwen.max_input_tokens,
                max_new_tokens=cfg.qwen.max_new_tokens, image_max_pixels=cfg.qwen.image_max_pixels,
                load_timeout=cfg.qwen.load_timeout, inference_timeout=cfg.qwen.inference_timeout)
            if cfg.coverage.policy == 'qwen_commands':
                worker = CommandPolicyWorker(worker)
        from omni_drones import init_simulation_app
        app = init_simulation_app(cfg)
        from omni_drones.envs.single.forest_survey import ForestSurvey
        env = ForestSurvey(cfg, cfg.headless)
        area_spec = cfg.coverage.get('work_area')
        work_area = (env.scene_info['bounds'] if isinstance(area_spec, str) and area_spec == 'map_bounds' else area_spec)
        if work_area is not None:
            work_area = np.asarray(work_area, float).tolist()
            public_area = dict(xy_bounds=work_area, coverage_scope='BOUND_PUBLIC_WORK_AREA',
                source='operator-declared existing map XY bounds; no terrain heights or tree geometry')
            write_json(output/'public_work_area.json', public_area)
            protocol.update(public_work_area=public_area,
                boundary_scope='observed absence inside the declared job; outside is outside scope, not observed free')
            write_json(output/'protocol.json', protocol)
        home = np.asarray(env.scene_info['start'] if cfg.coverage.home is None else cfg.coverage.home, float)
        initial_yaw = float(env.scene_info['initial_yaw'] if cfg.coverage.initial_yaw is None else cfg.coverage.initial_yaw)
        env.scene_info.update(start=home.tolist(), initial_yaw=initial_yaw,
            reference=[home.tolist(), home.tolist()], goal=home.tolist())
        env.reset()
        if env.enable_viewport:
            from .forest_viewer import ForestViewer
            viewer = ForestViewer(env, cfg.viewer, output)
            env.update_camera(force=True)
        write_json(output/'scene_truth.json', env.scene_info)
        np.savez_compressed(output/'scene_mesh.npz', vertices=env.mesh_vertices, faces=env.mesh_faces)
        structure = OnlineStructure(output, row_gap_bound=cfg.coverage.row_gap_bound,
            stem_gap_bound=cfg.coverage.stem_gap_bound, work_area_bounds=work_area,
            terrain_truth=terrain_truth)
        mission = CoverageMission(worker, cfg.coverage.instruction, home, initial_yaw,
            SimpleNamespace(**dict(cfg.task.flight)), output, cfg.coverage.operating_bounds,
            cfg.coverage.height_bounds, structure, command_file=cfg.coverage.command_file,
            decision_policy='qwen' if cfg.coverage.policy in ('qwen', 'teacher') else 'rule',
            decision_input=('point_choice' if cfg.coverage.policy == 'teacher'
                else cfg.coverage.get('decision_input', 'legacy')),
            decision_facts=cfg.coverage.get('decision_facts', False),
            decision_images=cfg.coverage.get('decision_images', 'both'),
            # The teacher answers by rule; candidate order does not matter to it.
            decision_orders=(int(cfg.coverage.get('decision_orders', 1)) if cfg.coverage.policy == 'qwen' else 1),
            planning_clearance_reserve=float(cfg.lower.get('planning_clearance_reserve', 0.)),
            desired_agl=cfg.coverage.get('desired_agl'), start_time=env.elapsed,
            goal_timeout=cfg.coverage.get('goal_timeout', 90.),
            no_motion_seconds=cfg.coverage.get('no_motion_seconds', 25.),
            no_progress_seconds=cfg.coverage.get('no_progress_seconds'),
            recovery_refill_goals=cfg.coverage.get('recovery_refill_goals'),
            max_climb_slope=cfg.coverage.get('max_climb_slope'),
            ground_fallback=cfg.coverage.get('ground_fallback'),
            terrain_truth=terrain_truth,
            coverage_stall_seconds=cfg.coverage.get('coverage_stall_seconds'),
            max_decisions=cfg.coverage.get('max_decisions', 160),
            max_recoveries=cfg.coverage.get('max_recoveries', 3),
            max_bootstrap_observations=cfg.coverage.get('max_bootstrap_observations', 32),
            max_rejections=cfg.coverage.get('max_rejections', 5), max_observation_age=cfg.local.max_observation_age)
        if viewer is not None:
            viewer.mission = mission
        mission.command_schedule = OmegaConf.to_container(cfg.coverage.command_schedule, resolve=True)
        xmin, xmax, ymin, ymax = cfg.coverage.operating_bounds
        guidance = Guidance(mission.mission_id, 1, mission.current_goal.goal_id, home[None],
            speed_limit=min(cfg.local.speed_limit, cfg.task.flight.max_speed),
            corridor=np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]),
            height_bounds=tuple(cfg.coverage.height_bounds))
        from .lower_flight import run_lower
        result = run_lower(env, cfg, output, traces, guidance, mission=mission)
        rows = np.asarray(traces).reshape(-1, 21)
        np.save(output/'trajectory.npy', rows)
        actual = np.vstack([home, rows[:, 8:11]])
        from .evaluation import truth_collision, sample_polyline
        from .geometry import INFLATION, VERTICAL_RADIUS
        violation = bool(truth_collision(sample_polyline(actual), dict(mesh=(env.mesh_vertices, env.mesh_faces)),
            INFLATION, VERTICAL_RADIUS))

        state = env.state[0, 0, :13].detach().cpu().numpy()
        if native:
            # Hidden scene geometry first enters task scoring after flight ends.
            from .native_coverage_evaluation import evaluate_native_coverage
            truth = dict(env.scene_info, evaluation_terrain_file=str((map_folder/'terrain.npz').resolve()),
                evaluation_protocol=dict(agl_bounds=list(cfg.coverage.agl_bounds), nominal_agl=cfg.coverage.desired_agl,
                    work_area_bounds=work_area))
            independent = evaluate_native_coverage(rows[:, 8:11], state[7:10], home,
                result.get('physical_stop_verified'), mission, violation, truth,
                trace=rows, decisions=output)
        else:
            evaluation_path = map_folder/'evaluation.json'
            evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
            independent = evaluate_coverage(actual, state[7:10], home, result.get('physical_stop_verified'),
                mission, violation, evaluation)
        from .coverage_command_audit import audit_coverage_commands
        lower_events = [json.loads(line) for line in (output/'decisions.jsonl').read_text().splitlines()]
        command_audit = audit_coverage_commands(mission.events, lower_events,
            OmegaConf.to_container(cfg.coverage.command_schedule, resolve=True), policy=cfg.coverage.policy)
        write_json(output/'command_audit.json', command_audit)
        # A teacher run has no model to prove; it passes on the flight task alone.
        verdict = ('task_passed_without_model_provenance' if cfg.coverage.policy == 'teacher'
            else 'full_coverage_passed')
        result.update(evaluation=independent, policy=cfg.coverage.policy, passed_scope=verdict,
            passed=bool(independent.get(verdict, independent['full_coverage_passed'])
                and result.get('reason') == 'mission_reached' and result.get('failure_reason') is None
                and (not cfg.coverage.command_schedule or command_audit['status'] == 'COMPLETE')),
            command_audit_status=command_audit['status'],
            simulated_seconds=float(env.elapsed), total_wall_seconds=time.perf_counter()-began,
            actual_path_length_m=float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum()),
            map_unchanged=all(file_hash(map_folder/name) == digest for name, digest in map_hashes.items()))
        result['passed'] = bool(result['passed'] and result['map_unchanged'])
        write_json(output/'task_evaluation.json', independent)
    except Exception as error:
        import traceback
        result.update(reason=type(error).__name__, error=str(error), traceback=traceback.format_exc(), passed=False)
        traceback.print_exc()
    finally:
        if structure is not None:
            structure.close()
        if mission is not None:
            write_json(output/'mission.json', mission.definition())
            write_json(output/'mission_events.json', mission.events)
            write_json(output/'goal_results.json', mission.results)
            mission.close()
        if traces:
            np.save(output/'trajectory.npy', np.asarray(traces).reshape(-1, 21))
        changed_sources = [name for name, digest in source_hashes.items()
            if not (root/name).is_file() or file_hash(root/name) != digest]
        result.update(source_unchanged=not changed_sources, changed_sources=changed_sources)
        result['passed'] = bool(result['passed'] and result['source_unchanged'])
        write_json(output/'result.json', result)
        print(f'COVERAGE_RESULT={result["reason"]} passed={result["passed"]}', flush=True)
        with ExitStack() as cleanup:
            if app is not None:
                cleanup.callback(app.close)
            if env is not None:
                cleanup.callback(env.close)
            if worker is not None:
                cleanup.callback(worker.close)
            if viewer is not None:
                viewer.save()
                cleanup.callback(viewer.hold_result, app, result)
    return result
