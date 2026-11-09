"""First zero-shot language/subgoal pilot, with measured feedback and no truth access.

This deliberately small pilot tests public takeoff-relative direction commands,
then return/stop. It does not claim discovered tree-row identities or inspection
coverage. The policy selects every movement goal online from sensed candidates;
an independent evaluator judges the requested task after physical execution.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import time

import numpy as np

from .subgoal import GoalMonitor, Subgoal
from .spline_runtime import observation_age


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n')


def policy_prompt(instruction, home, initial_yaw, position, yaw, candidates, history, last_rejection,
                  active_clause=None):
    """No answer, required target ID, teacher route or geometry truth is accepted."""
    compact_candidates = [dict(candidate_id=c['candidate_id'],
        relative_takeoff_xyz_m=c['relative_home_xyz_m'],
        relative_current_body_xyz_m=c['relative_body_xyz_m']) for c in candidates]
    completed_ids = [r['candidate_id'] for r in history]
    state = dict(home_world=list(home), position_world=np.round(position, 3).tolist(),
        yaw_world_rad=round(float(yaw), 3), takeoff_yaw_rad=float(initial_yaw),
        at_home=bool(np.linalg.norm(np.asarray(position)-home) <= .25),
        completed_candidate_ids_in_order=completed_ids,
        active_clause=active_clause, last_rejection=last_rejection, candidates=compact_candidates)
    return (
        '你是无人机任务决策器。严格按用户指令的顺序选择下一目标，每次只选一步。\n'
        '只输出一个JSON对象，恰好含candidate_id和behavior；不要解释，不要输出坐标或路线。\n'
        'completed_candidate_ids_in_order是程序根据实际到达并停稳生成的可信完成记录，'
        '不是建议，也不是待执行列表。先核对它，再决定指令中尚未完成的第一步；不要重复已完成动作。\n'
        'LEFT=起点左侧，RIGHT=起点右侧，FORWARD=起点前方，BACKWARD=起点后方；'
        '方向固定相对起飞朝向。HOME=起点，HOLD=保持当前位置。'
        '候选表包含干扰选项，图中可走的位置不一定是用户要求的位置。'
        '图中灰色未知、橙色阻挡、浅色已观测；飞行净空由下层复核。\n'
        '行为：前往方向点用FOLLOW，返回HOME用RETURN，任务完成用HOLD和REQUEST_STOP，'
        '所需目标暂不可观测时用HOLD和OBSERVE。\n'
        '任务推进示例（只说明如何使用历史，不是当前任务）：\n'
        '指令=先到起点后方再返回结束；完成记录=[]；输出={"candidate_id":"BACKWARD","behavior":"FOLLOW"}\n'
        '指令=先到起点前方再返回结束；完成记录=["FORWARD"]；输出={"candidate_id":"HOME","behavior":"RETURN"}\n'
        '指令=先到起点前方再返回结束；完成记录=["FORWARD","HOME"]；at_home=true；'
        '输出={"candidate_id":"HOLD","behavior":"REQUEST_STOP"}\n'
        '原始用户指令：'+instruction+'\n'
        '当前只执行active_clause指定的一个子任务；它来自Qwen对原始指令的解析，'
        '并由程序依据实测完成推进。VISIT选择direction对应候选；RETURN_HOME选择HOME；'
        'STOP选择HOLD和REQUEST_STOP。已经结束的子任务不得重新开始。\n'
        '当前信息：'+json.dumps(state, ensure_ascii=False, separators=(',', ':')))


class QwenTaskMission:
    """Duck-typed lower-layer mission with asynchronous online goal selection.

    Only ``advance`` after an independently measured handoff stop can submit or
    activate policy choices. Rendering and inference are outside the control
    thread. An initial stationary warmup is excluded from movement progress.
    ``COMPLETE`` means the model requested an end; task correctness is assessed
    separately against the original instruction's frozen evaluation predicates.
    """
    def __init__(self, worker, instruction, home, initial_yaw, flight, output,
                 operating_bounds, height_bounds, *, start_time=0.,
                 goal_timeout=45., no_motion_seconds=20., max_decisions=8,
                 max_rejections=3, step_distance=2., max_observation_age=.3,
                 prepare_fn=None, candidate_fn=None, task_plan=None):
        from .subgoal_inputs import build_candidates
        self.worker, self.instruction = worker, str(instruction)
        if not self.instruction.strip():
            raise ValueError('a nonempty language instruction is required')
        self.home = np.asarray(home, float)
        if self.home.shape != (3,) or not np.isfinite(self.home).all():
            raise ValueError('home must be finite world XYZ')
        if max_decisions < 1 or max_rejections < 1:
            raise ValueError('decision and rejection budgets must be positive')
        self.initial_yaw, self.flight = float(initial_yaw), flight
        self.output = Path(output)
        self.decisions_dir = self.output/'language_decisions'
        self.decisions_dir.mkdir(exist_ok=False)
        self.operating_bounds, self.height_bounds = tuple(operating_bounds), tuple(height_bounds)
        self.goal_timeout, self.no_motion_seconds = goal_timeout, no_motion_seconds
        self.max_decisions, self.max_rejections = max_decisions, max_rejections
        self.step_distance, self.max_observation_age = step_distance, max_observation_age
        self.start_time = self.last_time = float(start_time)
        self.mission_id, self.task_revision = 'qwen_subgoal_pilot_v1', 1
        self.task_plan = deepcopy(task_plan)
        self.clause_index = 0
        self.status, self.reason = 'RUNNING', ''
        self.goals = [Subgoal('sensor_warmup', 1, tuple(self.home), arrival_yaw=initial_yaw,
            position_tolerance=.2, speed_tolerance=.05, dwell_seconds=.5,
            timeout_seconds=goal_timeout, no_motion_seconds=no_motion_seconds)]
        self.index = 0
        self.monitor = GoalMonitor(self.current_goal, self.start_time)
        self._goal_feedback, self._results, self._events = None, [], []
        self.history, self.last_rejection = [], None
        self.latest, self.pending, self.preparation, self.validation = None, None, None, None
        self.decisions, self.rejections = 0, 0
        self.stop_requested, self._closed = False, False
        self._candidate_fn = candidate_fn or build_candidates
        self._prepare_fn = prepare_fn or self._prepare
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='upper-input')
        self._selected = None
        self._event('warmup_activated', start_time)

    @property
    def current_goal(self):
        return self.goals[self.index]

    @property
    def events(self):
        return deepcopy(self._events)

    @property
    def results(self):
        return deepcopy(self._results)

    def _event(self, kind, now, **details):
        row = dict(sequence=len(self._events), kind=kind, time=float(now),
                   task_revision=self.task_revision, goal_id=self.current_goal.goal_id,
                   revision=self.current_goal.revision, clause_index=self.clause_index, **details)
        self._events.append(row)
        with (self.output/'task_events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')

    def observe(self, snapshot):
        # Snapshot contains only lower-map points/ray evidence, timing and state.
        if self.status not in ('COMPLETE', 'FAILED'):
            self.latest = snapshot

    def accepts(self, goal_id, revision):
        return self.status == 'RUNNING' and goal_id == self.current_goal.goal_id and revision == self.current_goal.revision

    def guidance_points(self, actual_position):
        points = np.asarray([actual_position, self.current_goal.target_world], float)
        return points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]

    def feedback(self):
        return dict(mission_id=self.mission_id, status=self.status, reason=self.reason,
            goal_index=self.index, total_goals=len(self.goals),
            completed_goals=sum(r['status'] == 'COMPLETE' for r in self._results),
            completed_movement_goals=len(self.history), decisions=self.decisions,
            active_clause=self.active_clause,
            rejections=self.rejections, stop_requested=self.stop_requested,
            goal_id=self.current_goal.goal_id, revision=self.current_goal.revision,
            start_time=self.start_time, time=self.last_time,
            elapsed_seconds=self.last_time-self.start_time, goal_start_time=self.monitor.start_time,
            goal_feedback=deepcopy(self._goal_feedback),
            completion_semantics='policy end request; independent task predicates required')

    def _fail(self, reason, now):
        if self.status != 'FAILED':
            self.status, self.reason = 'FAILED', str(reason)
            self._event('task_failed', now, reason=self.reason)

    def update(self, now, position, velocity, yaw, failure=None):
        if not np.isfinite(np.r_[now, position, velocity, yaw]).all() or now < self.last_time:
            raise ValueError('finite actual state and monotonic time required')
        self.last_time = float(now)
        if self.status == 'FAILED':
            return self.feedback()
        if self.status == 'COMPLETE':
            if failure:
                self._fail(failure, now)
            return self.feedback()
        if self.status == 'WAITING_NEXT':
            if failure:
                self._fail(failure, now)
            return self.feedback()
        self._goal_feedback = self.monitor.update(now, position, velocity, yaw, failure)
        if failure:
            row = dict(self._goal_feedback, goal_index=self.index,
                target_world=list(self.current_goal.target_world),
                start_time=self.monitor.start_time, end_time=float(now))
            self._results.append(row)
            self._event('goal_failed', now, feedback=row,
                actual_position=np.asarray(position).tolist(), actual_velocity=np.asarray(velocity).tolist())
            self._fail(failure, now)
            return self.feedback()
        status = self._goal_feedback['status']
        if status in ('COMPLETE', 'FAILED'):
            row = dict(self._goal_feedback, goal_index=self.index,
                target_world=list(self.current_goal.target_world),
                start_time=self.monitor.start_time, end_time=float(now))
            self._results.append(row)
            self._event('goal_completed' if status == 'COMPLETE' else 'goal_failed', now,
                actual_position=np.asarray(position).tolist(), actual_velocity=np.asarray(velocity).tolist(),
                feedback=row)
            if status == 'FAILED':
                self._fail(self._goal_feedback['reason'], now)
            else:
                if self._selected is not None:
                    self.history.append(dict(candidate_id=self._selected['candidate_id'],
                        behavior=self._selected['behavior'], target_world=list(self.current_goal.target_world),
                        actual_position=np.round(position, 4).tolist(),
                        actual_speed_mps=float(np.linalg.norm(velocity)), completed_at=float(now)))
                    if self.task_plan is not None:
                        self.clause_index += 1
                self.status, self.reason = 'WAITING_NEXT', 'goal_reached'
        return self.feedback()

    @property
    def active_clause(self):
        return (None if self.task_plan is None else
                deepcopy(self.task_plan['steps'][self.clause_index]))

    def _candidates(self, snapshot, position, yaw):
        return self._candidate_fn(snapshot, position, yaw, self.home, self.flight,
            operating_bounds=self.operating_bounds, height_bounds=self.height_bounds,
            step_distance=self.step_distance, anchor_yaw=self.initial_yaw)

    def _prepare(self, snapshot, position, yaw, request, folder, history, last_rejection):
        from .subgoal_inputs import render_candidate_bev
        candidates = self._candidates(snapshot, position, yaw)
        image = folder/'input.png'
        render_candidate_bev(snapshot, position, yaw, self.home, candidates, image,
            self.flight, operating_bounds=self.operating_bounds, height_bounds=self.height_bounds,
            trail=[self.home.tolist()]+[r['actual_position'] for r in history], anchor_yaw=self.initial_yaw)
        prompt = policy_prompt(self.instruction, self.home, self.initial_yaw, position, yaw,
                               candidates, history, last_rejection, self.active_clause)
        payload = dict(request_id=request, task_revision=self.task_revision,
            goal_revision=self.current_goal.revision, instruction=self.instruction,
            source_time=float(snapshot['timestamp']), candidates=candidates,
            completed_movement_goals=history, prompt=prompt,
            active_clause=self.active_clause,
            position_world=position.tolist(), yaw=float(yaw))
        _write(folder/'input.json', payload)
        np.savez_compressed(folder/'observation.npz', points=snapshot['points'],
            evidence=snapshot['evidence'], position=position, yaw=yaw,
            timestamp=snapshot['timestamp'])
        return dict(payload=payload, image=str(image), prompt=prompt, candidates=candidates)

    def _reject(self, reason, now, details=None):
        self.rejections += 1
        self.last_rejection = dict(reason=reason, details=details)
        self._event('policy_rejected', now, reason=reason, details=details)
        self.pending = None
        if self.rejections >= self.max_rejections:
            self._fail('policy_rejection_budget', now)

    def _model_error(self, response, now):
        self._fail('model_error', now)

    def _submit_order(self, k):
        order = self.pending['orders'][k]
        return self.worker.submit(order['image'], order['prompt'], request_id=f"{self.pending['request_id']}_o{k}",
                                  response_schema='choice', letter_scores=True)

    def _order_reply(self, reply, now):
        """Keep one order's reply; ask the next order, or return the combined response.

        An error or a foreign request ID ends the decision as for one reply.
        """
        from .coverage_prompt import combined_choice
        pending = self.pending
        k = len(pending['order_replies'])
        _write(pending['folder']/f'response_o{k}.json', reply)
        if reply.get('error') or reply.get('request_id') != f"{pending['request_id']}_o{k}":
            return dict(reply, request_id=pending['request_id'] if reply.get('error') else reply.get('request_id'),
                        failed_order=k)
        pending['order_replies'].append(reply)
        if k+1 < len(pending['orders']):
            if not self._submit_order(k+1):
                self._reject('WORKER_BUSY', now)
            return None
        replies = pending['order_replies']
        pick, scores = combined_choice([(o['labels'], r) for o, r in zip(pending['orders'], replies)])
        return dict(request_id=pending['request_id'], response_schema='choice', orders=replies,
            combined=dict(candidate_id=pick, scores=scores,
                          rule='letter probabilities normalised per order, summed per candidate'),
            **{key: sum(r.get(key) or 0 for r in replies)
               for key in ('input_tokens', 'output_tokens', 'latency_seconds', 'wall_seconds')})

    def advance(self, now, position, velocity, yaw, reference_settled=True):
        if self._closed or self.status != 'WAITING_NEXT':
            return None
        self.last_time = float(now)
        position, velocity = np.asarray(position, float), np.asarray(velocity, float)
        if not reference_settled or np.linalg.norm(velocity) > .05:
            return None
        if self.preparation is not None:
            if not self.preparation.done():
                return None
            try:
                prepared = self.preparation.result()
                self.pending.update(prepared)
                if 'orders' in prepared:
                    self.pending['order_replies'] = []
                    submitted = self._submit_order(0)
                else:
                    submitted = self.worker.submit(prepared['image'], prepared['prompt'],
                        request_id=self.pending['request_id'],
                        **({'response_schema': prepared['response_schema']} if 'response_schema' in prepared else {}))
                if not submitted:
                    self._reject('WORKER_BUSY', now)
            except Exception as error:
                self._reject('INPUT_PREPARATION', now, f'{type(error).__name__}: {error}')
            self.preparation = None
        if self.status != 'WAITING_NEXT':
            return None
        # Once decoded, retain the response while waiting for fresh geometry.
        # Do not poll the already-drained model pipe again in that state.
        if self.pending is not None and 'ready_response' in self.pending:
            pending = self.pending
            response = pending['ready_response']
            decision = response['decision']
            selected = next(c for c in pending['candidates'] if c['candidate_id'] == decision['candidate_id'])
            if self.validation is not None:
                if not self.validation.done():
                    return None
                try:
                    valid, snapshot, checked_position = self.validation.result()
                except Exception as error:
                    self.validation = None
                    self._reject('REVALIDATION_ERROR', now, str(error))
                    return None
                self.validation = None
                if not valid:
                    self._reject('CANDIDATE_NO_LONGER_OBSERVED', now)
                    return None
                fresh = observation_age(now, time.perf_counter(), snapshot['timestamp'], snapshot['captured']) <= self.max_observation_age
                if fresh and np.linalg.norm(position-checked_position) <= .05:
                    return self._activate(decision, selected, pending, now, position, velocity, yaw,
                                          checked_snapshot=snapshot)
            if self.latest is not None and observation_age(now, time.perf_counter(),
                    self.latest['timestamp'], self.latest['captured']) <= self.max_observation_age:
                self.validation = self._pool.submit(self._validate, self.latest, position.copy(), float(yaw), selected)
            return None
        if self.pending is not None and 'candidates' in self.pending:
            response = self.worker.poll()
            if response is None:
                return None
            pending = self.pending
            if 'order_replies' in pending:
                response = self._order_reply(response, now)
                if response is None:
                    return None
            _write(pending['folder']/'response.json', response)
            self._event('policy_response', now, request_id=pending['request_id'], response=response)
            if response.get('request_id') != pending['request_id']:
                self._reject('REQUEST_ID', now)
                return None
            if response.get('error'):
                self._model_error(response, now)
                return None
            if pending['goal_revision'] != self.current_goal.revision or pending['task_revision'] != self.task_revision:
                self._reject('SUPERSEDED', now)
                return None
            decision = response.get('decision')
            if pending.get('response_schema') == 'choice':
                # A letter names one offered candidate; behaviour follows its role.
                from .coverage_prompt import decision_from_choice
                if 'combined' in response:
                    choice = pending['labels'].get(response['combined']['candidate_id'])
                    response['next_aisle'] = [(r.get('choice') or {}).get('next_aisle') for r in response['orders']]
                else:
                    choice = (response.get('choice') or {}).get('choice')
                    response['next_aisle'] = (response.get('choice') or {}).get('next_aisle')
                decision = decision_from_choice(choice, pending['labels'], pending['candidates'])
                if decision is None:
                    self._reject('UNKNOWN_CANDIDATE', now, f'choice {choice!r} is not an offered letter')
                    return None
                response['decision'] = decision
            if not isinstance(decision, dict) or set(decision) != {'candidate_id', 'behavior'}:
                self._reject('OUTPUT_SCHEMA', now)
                return None
            matches = [c for c in pending['candidates'] if c['candidate_id'] == decision['candidate_id']]
            if len(matches) != 1:
                self._reject('UNKNOWN_CANDIDATE', now)
                return None
            if decision['behavior'] == 'REQUEST_STOP':
                if hasattr(self, 'allows_stop') and not self.allows_stop():
                    self._reject('TASK_NOT_FINISHED', now)
                    return None
                if decision['candidate_id'] != 'HOLD':
                    self._reject('STOP_REQUIRES_HOLD', now)
                    return None
                if self.task_plan is not None and self.active_clause['action'] != 'STOP':
                    self._reject('TASK_CLAUSE_NOT_COMPLETE', now)
                    return None
                self.stop_requested = True
                self.status, self.reason = 'COMPLETE', 'policy_finished'
                self.pending = None
                self._event('policy_stop_requested', now, actual_position=position.tolist(),
                            actual_speed_mps=float(np.linalg.norm(velocity)))
                return None
            if decision['candidate_id'] == 'HOLD':
                self._reject('OBSERVE_RETRY', now, 'stationary scanning continues; no mission progress credited')
                return None
            if decision['behavior'] not in ('FOLLOW', 'CHANGE_AISLE', 'REJOIN', 'BACKTRACK', 'RETURN'):
                self._reject('INVALID_MOVEMENT_BEHAVIOR', now)
                return None
            # Selected-target geometry checks stay outside the control loop.
            # The semantic decision may outlive its source frame; a separate
            # fresh observation must authorize its eventual movement target.
            pending['ready_response'] = response
            return None
        if self.pending is None:
            if self.decisions >= self.max_decisions:
                self._fail('policy_decision_budget', now)
                return None
            if self.latest is None or observation_age(now, time.perf_counter(),
                    self.latest['timestamp'], self.latest['captured']) > self.max_observation_age:
                return None
            request = f'decision_{self.decisions:03d}'
            folder = self.decisions_dir/request
            folder.mkdir()
            self.pending = dict(request_id=request, folder=folder,
                task_revision=self.task_revision, goal_revision=self.current_goal.revision)
            self.decisions += 1
            self._event('policy_requested', now, request_id=request,
                        source_time=float(self.latest['timestamp']))
            self.preparation = self._pool.submit(self._prepare_fn, self.latest,
                position.copy(), float(yaw), request, folder, deepcopy(self.history), deepcopy(self.last_rejection))
        return None

    def _validate(self, snapshot, position, yaw, selected):
        from .subgoal_inputs import validate_candidate
        valid = validate_candidate(snapshot, position, yaw, selected, self.flight,
            operating_bounds=self.operating_bounds, height_bounds=self.height_bounds)
        return valid, snapshot, position

    def _activate(self, decision, selected, pending, now, position, velocity, yaw, checked_snapshot):
        if (self.status != 'WAITING_NEXT' or pending['task_revision'] != self.task_revision
                or pending['goal_revision'] != self.current_goal.revision):
            self._reject('SUPERSEDED', now)
            return None
        if self.task_plan is not None:
            clause = self.active_clause
            required = clause.get('direction') if clause['action'] == 'VISIT' else 'HOME'
            if clause['action'] == 'STOP' or decision['candidate_id'] != required:
                self._reject('TASK_CLAUSE_MISMATCH', now, dict(active_clause=clause,
                                                             selected=decision['candidate_id']))
                return None
        target = np.asarray(selected['target_world'], float)
        if np.linalg.norm(target-position) < .3:
            self._reject('ALREADY_AT_TARGET', now)
            return None
        revision = self.current_goal.revision+1
        goal = Subgoal(f'qwen_{revision:03d}_{decision["candidate_id"]}', revision, tuple(target),
            arrival_yaw=self.initial_yaw, position_tolerance=.2, speed_tolerance=.05,
            dwell_seconds=.5, timeout_seconds=self.goal_timeout,
            no_motion_seconds=self.no_motion_seconds)
        self.goals.append(goal)
        self.index += 1
        self.monitor = GoalMonitor(goal, now)
        self._goal_feedback = self.monitor.update(now, position, velocity, yaw)
        self._selected = deepcopy(decision)
        self.status, self.reason = 'RUNNING', ''
        self.pending = None
        self._event('goal_activated', now, decision=decision, request_id=pending['request_id'],
            target_world=target.tolist(), revalidated_observation_time=float(checked_snapshot['timestamp']),
            actual_position=position.tolist(), actual_velocity=velocity.tolist())
        return goal

    def definition(self):
        return dict(protocol='qwen_language_subgoal_pilot_v1', mission_id=self.mission_id,
            instruction=self.instruction, start=self.home.tolist(), initial_yaw=self.initial_yaw,
            compiled_task_plan=self.task_plan,
            operating_bounds=list(self.operating_bounds), height_bounds=list(self.height_bounds),
            goals=[asdict(g) for g in self.goals],
            task_guidance='movement goals selected by actual pretrained Qwen responses from online observed candidates',
            reset_policy='one reset before warmup; no resets or teleports at policy decisions',
            limitations='takeoff-relative directional pilot; no row discovery, coverage or AGL claim')

    def close(self):
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)


def evaluate_direction_task(task_case, home, initial_yaw, distance, goal_results, final_position,
                            final_velocity, physical_stop_verified):
    """Independent frozen pilot predicates; NEVER passed to the policy supervisor."""
    offsets = dict(left_return=(0., 1.), right_return=(0., -1.), forward_return=(1., 0.))
    if task_case not in offsets:
        raise ValueError('unknown pilot evaluation case')
    x, y = offsets[task_case]
    c, s = np.cos(initial_yaw), np.sin(initial_yaw)
    required = np.asarray(home)+distance*np.array([c*x-s*y, s*x+c*y, 0.])
    completed = [r for r in goal_results if r.get('status') == 'COMPLETE' and r.get('goal_id') != 'sensor_warmup']
    visit_index = next((i for i,r in enumerate(completed)
        if np.linalg.norm(np.asarray(r['target_world'])-required) <= .25), None)
    returned_after = visit_index is not None and any(
        np.linalg.norm(np.asarray(r['target_world'])-home) <= .25 for r in completed[visit_index+1:])
    sequence_correct = bool(len(completed) == 2 and visit_index == 0
        and np.linalg.norm(np.asarray(completed[1]['target_world'])-home) <= .25)
    final_home = bool(np.linalg.norm(np.asarray(final_position)-home) <= .25)
    final_quiet = bool(np.linalg.norm(final_velocity) <= .05 and physical_stop_verified)
    return dict(task_case=task_case, required_visit_world=required.tolist(),
        required_visit_completed=visit_index is not None, returned_after_required_visit=bool(returned_after),
        movement_sequence_correct=sequence_correct,
        final_home=final_home, final_measured_stop=final_quiet,
        passed=bool(sequence_correct and returned_after and final_home and final_quiet))
