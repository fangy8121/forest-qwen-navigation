"""Online coverage, versioned language interruption and measured resumption.

All geometry enters through online structure/safety snapshots. No scene manifest,
teacher trajectory, true row count or evaluation answer is accepted here.
"""
from copy import copy, deepcopy
import json
from pathlib import Path
import time
import numpy as np
from .coverage import CoverageLedger, completion_evidence
from .geometry import world_to_local
from .subgoal import GoalMonitor, Subgoal
from .task_supervisor import QwenTaskMission, _write


class CoverageMission(QwenTaskMission):
    def __init__(self, worker, instruction, home, yaw, flight, output, bounds, heights,
                 structure, *, command_file=None, decision_policy='rule', desired_agl=None,
                 max_recoveries=3, max_bootstrap_observations=32, planning_clearance_reserve=0.,
                 no_progress_seconds=None, recovery_refill_goals=None, max_climb_slope=None,
                 ground_fallback=None, terrain_truth=None, coverage_stall_seconds=None,
                 decision_input='legacy', decision_facts=False, decision_images='both', decision_orders=1,
                 **kwargs):
        if decision_policy not in ('rule', 'qwen'):
            raise ValueError('decision_policy must be rule or qwen')
        if decision_input not in ('legacy', 'point_choice'):
            raise ValueError('decision_input must be legacy or point_choice')
        if decision_images not in ('both', 'global'):
            raise ValueError('decision_images must be both or global')
        if not isinstance(decision_orders, int) or decision_orders < 1:
            raise ValueError('decision_orders must be a positive integer')
        if desired_agl is not None and (not np.isfinite(desired_agl) or desired_agl <= 0):
            raise ValueError('desired_agl must be positive when enabled')
        if type(max_recoveries) is not int or max_recoveries < 0:
            raise ValueError('max_recoveries must be a nonnegative integer')
        if type(max_bootstrap_observations) is not int or max_bootstrap_observations < 1:
            raise ValueError('max_bootstrap_observations must be a positive integer')
        if not np.isfinite(planning_clearance_reserve) or planning_clearance_reserve < 0:
            raise ValueError('planning_clearance_reserve must be finite and nonnegative')
        if no_progress_seconds is not None and (not np.isfinite(no_progress_seconds) or no_progress_seconds <= 0):
            raise ValueError('no_progress_seconds must be positive when enabled')
        if coverage_stall_seconds is not None and (not np.isfinite(coverage_stall_seconds) or coverage_stall_seconds <= 0):
            raise ValueError('coverage_stall_seconds must be positive when enabled')
        # Fail fast when coverage stops growing (run fd5c0103 shuttled between
        # two finished aisles for 1745 s until the episode timeout).
        self.coverage_stall_seconds = None if coverage_stall_seconds is None else float(coverage_stall_seconds)
        self._coverage_progress, self._coverage_progress_time = None, None
        if ground_fallback not in (None, 'hold_current'):
            raise ValueError("ground_fallback must be None or 'hold_current'")
        # catkin_ws commander: a waypoint keeps the current altitude unless the
        # ground-height service returns a valid height. A level upright MID-360
        # at 2 m AGL observes almost no ground within ~17 m (run 0c23d8ef), so
        # requiring observed ground rejected every move. Obstacle and observed-
        # space checks are unchanged; ground hits remain obstacles.
        self.ground_fallback, self.ground_fallback_points = ground_fallback, 0
        # Simulation stand-in for an airframe that measures its own AGL (the
        # deployment M300 RTK): ground heights come from the map's known
        # terrain grid, not from LiDAR. The LiDAR still does all obstacle work.
        if terrain_truth is not None and set(terrain_truth) != {'xs', 'ys', 'heights'}:
            raise ValueError('terrain_truth needs xs, ys and heights')
        self.terrain_truth = terrain_truth
        if max_climb_slope is not None and (not np.isfinite(max_climb_slope) or max_climb_slope <= 0):
            raise ValueError('max_climb_slope must be positive when enabled')
        self.max_climb_slope = None if max_climb_slope is None else float(max_climb_slope)
        if recovery_refill_goals is not None and (type(recovery_refill_goals) is not int or recovery_refill_goals < 1):
            raise ValueError('recovery_refill_goals must be a positive integer when enabled')
        # A fixed budget for a whole multi-hour mission is spent by unrelated early
        # blocks (run e1f4e618: 3 at bootstrap, then fatal at 2879 s). Restore it
        # only after this many consecutive completed goals since the last recovery.
        self.recovery_refill_goals, self._completed_since_recovery = recovery_refill_goals, 0
        self.decision_policy, self.desired_agl = decision_policy, desired_agl
        # point_choice: Qwen picks a lettered point on two images, without task-order
        # pruning (the rule order is logged as a reference). Rule policy keeps legacy.
        self.decision_input = decision_input if decision_policy == 'qwen' else 'legacy'
        # Decision bench (outputs/03_Qwen点选决策/测试集): aisle facts with the global image
        # alone scored 94.5% against 72.7% for both images without facts.
        self.decision_facts, self.decision_images = bool(decision_facts), decision_images
        # Ask each point choice in this many rotated candidate orders and sum the
        # letter probabilities (coverage_prompt.combined_choice).
        self.decision_orders = decision_orders
        self.rule_reference = None
        self.no_progress_seconds = None if no_progress_seconds is None else float(no_progress_seconds)
        self.planning_clearance_reserve = float(planning_clearance_reserve)
        super().__init__(worker, instruction, home, yaw, flight, output, bounds, heights,
            prepare_fn=self._prepare, **kwargs)
        self.mission_id = 'online_forest_coverage_v1'
        self.structure = structure
        self.ledger = CoverageLedger()
        self.intent, self.command_ready = 'COVER', False
        # Task specification bound from language: which constraints apply.
        # order S|free and an optional lateral-index subset for COVER; a
        # resolved target corridor for VISIT_CORRIDOR.
        self.task_spec = dict(order='S', corridors=None, target=None, target_corridor=None)
        self.command_text, self.command_revision = instruction, 1
        self.command_memory = None
        self.command_pending, self.switching = None, False
        self.command_file = Path(command_file) if command_file else self.output/'commands.jsonl'
        self.command_file.touch(exist_ok=True)
        self.command_offset, self.last_command_poll = 0, -np.inf
        self.seen_commands = set()
        self.command_received_at = {}
        self.command_results, self.suspended = [], []
        self.active_corridor, self.departed_corridor = None, None
        self.entered_corridors, self.breadcrumbs = [], [np.asarray(home, float).tolist()]
        self.route, self.probes, self.active_command = [], {}, 'initial'
        self.resume_after, self.preferred_corridor = False, None
        self.resume_context, self.temporary_corridor = None, None
        self.handoff_reason = 'goal_reached'
        self.coverage_result = 'IN_PROGRESS'
        self.automatic_return = False
        self.latest_position, self.latest_speed = np.asarray(home, float), 0.
        self.latest_yaw = float(yaw)
        self._consumed_results = 0
        self.command_schedule = []
        self.scheduled_sent = set()
        self.choices, self.commitment, self.continuation = [], None, None
        self.last_completed_intention = None
        self.route_request_id, self.route_candidate_id = None, None
        self.next_policy_time, self.hold_started = -np.inf, None
        self.max_recoveries, self.recoveries = max_recoveries, 0
        self.max_bootstrap_observations = max_bootstrap_observations
        self.requires_handoff_stop, self.recovery_pending = False, False
        self.structure_ambiguous = False
        self.blocked_goal, self._monitor_recovery_allowed = None, False

    def observe_frame(self, frame):
        self.structure.submit(frame)

    def _context(self):
        return deepcopy(dict(intent=self.intent, corridor_id=self.active_corridor,
            preferred_corridor=self.preferred_corridor, resume_after=self.resume_after,
            resume_context=self.resume_context, temporary_corridor=self.temporary_corridor,
            automatic_return=self.automatic_return, task_revision=self.task_revision,
            commitment=self.commitment, last_completed_intention=self.last_completed_intention,
            task_spec=self.task_spec))

    def _restore_context(self, context):
        context = context or dict(intent='COVER')
        self.intent = context['intent']
        self.preferred_corridor = context.get('preferred_corridor')
        if self.intent == 'COVER' and self.preferred_corridor is None:
            self.preferred_corridor = context.get('corridor_id')
        self.resume_after = context.get('resume_after', False)
        self.resume_context = deepcopy(context.get('resume_context'))
        self.temporary_corridor = context.get('temporary_corridor')
        self.automatic_return = context.get('automatic_return', False)
        self.commitment = deepcopy(context.get('commitment'))
        self.last_completed_intention = deepcopy(context.get('last_completed_intention'))
        self.task_spec = deepcopy(context.get('task_spec') or dict(order='S', corridors=None,
            target=None, target_corridor=None))
        self.choices, self.continuation = [], None
        self.route = []  # Recompute geometry; a saved route is not a new permit.

    def receive(self, command_id, text, now):
        if command_id in self.seen_commands or self.status in ('FAILED', 'COMPLETE'):
            return False
        if not isinstance(text, str) or not text.strip():
            return False
        self.seen_commands.add(command_id)
        self.command_received_at[command_id] = float(now)
        self.suspended.append(self._context())
        self.task_revision += 1
        self.command_revision = self.task_revision
        self.active_command, self.command_text = command_id, text
        self.command_memory = dict(corridor_ids=[c['id'] for c in self.structure.snapshot().get('corridors', [])],
            entered_in_order=list(self.entered_corridors), current_corridor=self.active_corridor,
            last_departed_corridor=self.departed_corridor,
            reference_time=float(now), reference_position=self.latest_position.tolist())
        self.automatic_return = False
        self.command_ready, self.switching = False, True
        self.route = []
        self.commitment, self.continuation, self.choices = None, None, []
        self.route_request_id, self.route_candidate_id = None, None
        self.handoff_reason = 'runtime_command:'+command_id
        if self.status == 'RUNNING':
            self._results.append(dict(goal_id=self.current_goal.goal_id,
                revision=self.current_goal.revision, status='CANCELLED', reason='runtime_command',
                target_world=list(self.current_goal.target_world), end_time=float(now)))
        self.status, self.reason = 'WAITING_NEXT', 'runtime_command'
        self._event('command_received', now, command_id=command_id, text=text,
            actual_position=self.latest_position.tolist(), actual_speed_mps=self.latest_speed)
        # Exact pause vocabulary is the fast path. Other meanings are compiled
        # by Qwen while the existing lower layer performs a measured handoff stop.
        if text.strip().lower() in ('暂停', '停一下', 'pause', 'stop'):
            self.intent, self.command_ready = 'PAUSE', True
            self.command_results.append(dict(command_id=command_id, intent='PAUSE', accepted=True))
        return True

    def _read_commands(self, now):
        if now-self.last_command_poll < .2:
            return
        self.last_command_poll = now
        with self.command_file.open('rb') as stream:
            stream.seek(self.command_offset)
            # A partial append is left unread until its newline arrives.
            while True:
                start = stream.tell()
                line = stream.readline(16385)
                if not line:
                    break
                if len(line) > 16384 or not line.endswith(b'\n'):
                    stream.seek(start)
                    break
                self.command_offset = stream.tell()
                try:
                    obj = json.loads(line)
                    self.receive(str(obj['command_id']), obj['text'], now)
                except (ValueError, KeyError, TypeError, UnicodeError) as error:
                    self._event('command_rejected', now, reason=str(error))

    def update(self, now, position, velocity, yaw, failure=None):
        self.latest_position, self.latest_speed = np.asarray(position).copy(), float(np.linalg.norm(velocity))
        self.latest_yaw = float(yaw)
        for command in self.command_schedule:
            after = command.get('after_command_id')
            if after and (after not in self.command_received_at or
                    now < self.command_received_at[after]+command.get('delay', 0.)):
                continue
            if (command['command_id'] not in self.scheduled_sent and now >= command['at']
                    and self.latest_speed >= command.get('min_speed', 0.)
                    and len(self.entered_corridors) >= command.get('min_corridors', 0)):
                with self.command_file.open('a') as stream:
                    stream.write(json.dumps(dict(command_id=command['command_id'], text=command['text']), ensure_ascii=False)+'\n')
                self.scheduled_sent.add(command['command_id'])
        self._read_commands(now)
        if self.structure.last_error and self.status not in ('COMPLETE', 'FAILED'):
            self._fail('STRUCTURE:'+self.structure.last_error, now)
        observed = self.structure.snapshot()
        corridors = observed.get('corridors', [])
        if (self.status != 'FAILED' and failure is None
                and self.coverage_result == 'COMPLETE_UNDER_ASSUMPTIONS'
                and not self._completion(observed)['eligible']):
            self.coverage_result = 'IN_PROGRESS'
            self._event('coverage_completion_revoked', now)
            if self.automatic_return or self.intent == 'FINISH' or self.stop_requested:
                if self.status == 'RUNNING':
                    self._results.append(dict(goal_id=self.current_goal.goal_id,
                        revision=self.current_goal.revision, status='CANCELLED', reason='coverage_reopened',
                        target_world=list(self.current_goal.target_world), end_time=float(now)))
                self.task_revision += 1
                self.intent, self.route, self.switching = 'COVER', [], True
                self.command_ready, self.automatic_return = True, False
                self.status, self.reason = 'WAITING_NEXT', 'coverage_reopened'
                self.handoff_reason = 'coverage_reopened'
                self.stop_requested = False
                self.commitment, self.continuation, self.choices = None, None, []
                self.route_request_id, self.route_candidate_id = None, None
        current = [c['id'] for c in corridors if abs(np.asarray(position)[:2]@c['normal']-c['offset_m']) < .9
            and c['span'][0]-.5 <= np.asarray(position)[:2]@c['direction'] <= c['span'][1]+.5]
        if len(current) == 1 and current[0] != self.active_corridor:
            self.departed_corridor = self.active_corridor or self.departed_corridor
            self.active_corridor = current[0]
            if current[0] not in self.entered_corridors:
                self.entered_corridors.append(current[0])
            self._event('corridor_entered', now, corridor_id=current[0],
                previous_corridor=self.departed_corridor, actual_position=np.asarray(position).tolist())
        inspection_roles = ('INSPECT', 'OBSERVE', 'BOOTSTRAP', 'TRANSIT') if self.decision_policy == 'qwen' else ('INSPECT',)
        # The stop after an inspection goal is measured motion too. Dropping it
        # until the next goal started left a 1.0 m interior gap, and Qwen flew
        # the whole 69 m aisle again for it (run c1c3147f).
        inspect = bool(self._selected and self._selected.get('role') in inspection_roles
            and self.status in ('RUNNING', 'WAITING_NEXT') and self.intent in ('COVER', 'VISIT_CORRIDOR'))
        from .spline_runtime import observation_age
        fresh = self.latest is not None and observation_age(now, time.perf_counter(),
            self.latest['timestamp'], self.latest['captured']) <= self.max_observation_age
        self.ledger.observe(now, position, corridors, enabled=inspect, sensor_valid=fresh)
        if (self.coverage_stall_seconds is not None and self.intent in ('COVER', 'VISIT_CORRIDOR')
                and self.status == 'RUNNING'):
            progress = (round(sum(b-a for intervals in self.ledger.covered.values() for a, b in intervals), 1),
                len(corridors), sum(bool(c['ends'][side].get('confirmed')) for c in corridors for side in c['ends']))
            if self.intent == 'VISIT_CORRIDOR':
                # Approaching the requested aisle is progress too.
                target = self._visit_target(corridors)
                lateral = self._lateral(corridors)
                here = lateral.get(self.active_corridor, {}).get('lateral_index')
                there = lateral.get(target, {}).get('lateral_index')
                progress += (-abs(here-there) if here is not None and there is not None else -99,)
            if self._coverage_progress is None or progress > self._coverage_progress:
                self._coverage_progress, self._coverage_progress_time = progress, float(now)
            elif now-self._coverage_progress_time > self.coverage_stall_seconds:
                self._event('coverage_stalled', now, progress=list(progress),
                    since=self._coverage_progress_time)
                super()._fail('coverage_stalled', now)
        if (self.status == 'RUNNING' and
                np.linalg.norm(np.asarray(position)-self.breadcrumbs[-1]) >= 1.):
            self.breadcrumbs.append(np.asarray(position).tolist())
        self._monitor_recovery_allowed = failure is None and self.status == 'RUNNING'
        try:
            result = super().update(now, position, velocity, yaw, failure)
        finally:
            self._monitor_recovery_allowed = False
        while self._consumed_results < len(self._results):
            row = self._results[self._consumed_results]
            self._consumed_results += 1
            if row['status'] == 'COMPLETE' and row['goal_id'] != 'sensor_warmup':
                if self.route:
                    self.route.pop(0)
                self._event('coverage_checkpoint', now, ledger=self.ledger.snapshot(corridors))
                self._completed_since_recovery += 1
                if (self.recovery_refill_goals is not None and self.recoveries
                        and self._completed_since_recovery >= self.recovery_refill_goals):
                    self._event('coverage_recovery_budget_restored', now, recoveries_used=self.recoveries,
                        completed_goals_since_recovery=self._completed_since_recovery)
                    self.recoveries = 0
        return result

    def _command_tick(self, now):
        if self.switching:
            if self.preparation is not None:
                if not self.preparation.done():
                    return False
                self.preparation = None
            if self.validation is not None:
                if not self.validation.done():
                    return False
                self.validation = None
            if self.worker.busy:
                stale = self.worker.poll()
                if stale is None:
                    return False
                self._event('superseded_model_response', now, response=stale)
            self.pending, self.command_pending, self.switching = None, None, False
        if self.command_ready:
            return True
        if self.command_pending is None:
            observed = self.structure.snapshot().get('corridors', [])
            order = self._lateral(observed)
            uncovered = {row['corridor_id']:row['uncovered_length_m']
                         for row in self.ledger.snapshot(observed)['corridors']}
            memory = deepcopy(self.command_memory) if self.command_memory is not None else dict(
                corridor_ids=[c['id'] for c in observed],
                corridors=sorted([dict(corridor_id=c['id'], lateral_index=order.get(c['id'], {}).get('lateral_index'),
                    uncovered_m=uncovered.get(c['id']), entered=c['id'] in self.entered_corridors)
                    for c in observed], key=lambda row:row['lateral_index'] or 0),
                current_lateral_index=order.get(self.active_corridor, {}).get('lateral_index'),
                entered_in_order=self.entered_corridors, current_corridor=self.active_corridor,
                last_departed_corridor=self.departed_corridor)
            memory.update(default_completion='return HOME and measured stop', command=self.command_text)
            request = f'command_{self.task_revision}_{len(self.command_results)}'
            self.worker.submit(None, json.dumps(memory, ensure_ascii=False), request_id=request,
                response_schema='coverage_command')
            self.command_pending = dict(request_id=request, task_revision=self.task_revision)
            return False
        response = self.worker.poll()
        if response is None:
            return False
        pending, self.command_pending = self.command_pending, None
        self._event('command_model_response', now, response=response)
        if response.get('error'):
            if not str(response['error']).startswith('MODEL_OUTPUT'):
                self._fail(response['error'], now)
                return False
            # A malformed interpretation must not end the mission: ask again
            # once, then reject only this command and keep the current task.
            self._command_attempts = getattr(self, '_command_attempts', 0)+1
            if self._command_attempts < 2:
                self._event('command_retry', now, command_id=self.active_command, error=response['error'])
                return False
            self._command_attempts = 0
            previous = next((c for c in reversed(self.suspended)
                if c['intent'] not in ('PAUSE', 'FINISH')), None)
            self._restore_context(previous)
            self.command_ready = True
            self.command_results.append(dict(command_id=self.active_command, accepted=False, error=response['error']))
            self._event('command_rejected', now, command_id=self.active_command, reason=response['error'],
                kept_intent=self.intent)
            return True
        self._command_attempts = 0
        if pending['task_revision'] != self.task_revision:
            return False
        command = response['coverage_command']
        if command['intent'] == 'REVISIT_CORRIDOR' and command['corridor_id'] not in self.entered_corridors:
            self._event('command_unresolved', now, command=command, reason='not a previously entered corridor')
            self.intent, self.command_ready = 'PAUSE', True
            return False
        previous = next((c for c in reversed(self.suspended)
            if c['intent'] not in ('PAUSE', 'FINISH')), None)
        if command['intent'] == 'RESUME':
            self._restore_context(previous)
        else:
            spec = deepcopy(self.task_spec)
            if command['intent'] == 'COVER':
                corridors = command.get('corridors')
                observed = len(self.structure.snapshot().get('corridors', []))
                if corridors and sorted(corridors) == list(range(1, observed+1)):
                    # Listing every observed aisle means all aisles; more may
                    # still be discovered, so keep the subset open.
                    self._event('command_normalized', now, command_id=self.active_command,
                        corridors=corridors, normalized_to=None)
                    corridors = None
                spec = dict(order=command.get('order') or 'S', corridors=corridors,
                            target=None, target_corridor=None)
            elif command['intent'] == 'VISIT_CORRIDOR':
                target = dict(command['target'])
                if 'relative' in target:
                    # Bind a heading-relative reference now; the heading changes.
                    index = self._relative_index(target)
                    if index is None:
                        self._event('command_unresolved', now, command=command,
                            reason='no current corridor for a relative reference')
                        self.intent, self.command_ready = 'PAUSE', True
                        return False
                    target = dict(lateral_index=index)
                spec = dict(spec, target=target, target_corridor=None)
            self.intent = command['intent']
            self.resume_after, self.preferred_corridor = command['resume_after'], command['corridor_id']
            self.resume_context = deepcopy(previous) if self.resume_after else None
            self.temporary_corridor = None
            self.task_spec = spec
        self.command_ready = True
        self.command_results.append(dict(command_id=self.active_command, **command, accepted=True))
        self._event('command_bound', now, command_id=self.active_command, command=command,
            actual_position=self.latest_position.tolist(), actual_speed_mps=self.latest_speed)
        return True

    def _lateral(self, corridors):
        from .coverage_candidates import corridor_lateral_order
        return corridor_lateral_order(corridors, self.home[:2])

    def _relative_index(self, target):
        """Lateral index left/right of the current heading, counted in aisles."""
        corridors = self.structure.snapshot().get('corridors', [])
        order = self._lateral(corridors)
        here = order.get(self.active_corridor, {}).get('lateral_index')
        if here is None or not corridors:
            return None
        from .coverage_candidates import lateral_reference
        reference = lateral_reference(corridors, self.home[:2])
        right = np.array([np.sin(self.latest_yaw), -np.cos(self.latest_yaw)])
        sign = 1 if right@reference >= 0 else -1
        if target['relative'] == 'left':
            sign = -sign
        index = here+sign*target['count']
        return index if index >= 1 else None

    def _spec_corridor_ids(self, corridors):
        """Observed corridor IDs required by the current COVER subset (None = all)."""
        wanted = self.task_spec.get('corridors')
        if not wanted:
            return None
        order = self._lateral(corridors)
        return {cid for cid, row in order.items() if row['lateral_index'] in wanted}

    def _visit_target(self, corridors):
        target = self.task_spec.get('target') or {}
        ids = {c['id'] for c in corridors}
        if target.get('corridor_id') in ids:
            return target['corridor_id']
        if 'lateral_index' in target:
            return next((cid for cid, row in self._lateral(corridors).items()
                         if row['lateral_index'] == target['lateral_index']), None)
        return None

    def _completion(self, structure):
        """Completion of the bound task: the whole stand, or the requested subset."""
        evidence = completion_evidence(structure, self.ledger)
        corridors = structure.get('corridors', [])
        wanted = self.task_spec.get('corridors')
        if not wanted:
            return evidence
        order = self._lateral(corridors)
        chosen = [c for c in corridors if order.get(c['id'], {}).get('lateral_index') in wanted]
        unresolved = [f'LATERAL_{i}:NOT_OBSERVED' for i in wanted
                      if i not in {order.get(c['id'], {}).get('lateral_index') for c in chosen}]
        unresolved += [c['id']+':UNCOVERED' for c in chosen if self.ledger.gaps(c)]
        unresolved += [c['id']+':'+side+':END_UNCONFIRMED' for c in chosen
                       for side in ('low', 'high') if not c['ends'][side].get('confirmed')]
        if structure.get('identity_ambiguous'):
            unresolved.append('IDENTITY_AMBIGUOUS')
        return dict(evidence, eligible=not unresolved, unresolved=unresolved,
            status='COMPLETE_UNDER_ASSUMPTIONS' if not unresolved else 'UNRESOLVED',
            requested_lateral_indices=list(wanted), coverage_scope='REQUESTED_CORRIDOR_SUBSET')

    def _ground(self, xy):
        """Ground height and validity at XY: known terrain if given, else online."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.terrain_truth is not None:
            from forest_generator.terrain import grid_height
            t = self.terrain_truth
            return grid_height(xy, t['xs'], t['ys'], t['heights']), np.ones(len(xy), bool)
        from .observed_terrain import query_terrain
        return query_terrain(self.structure.snapshot().get('terrain', {}), xy)

    def _anchor(self, xy, role, corridor_id=None):
        height = float(self.home[2])
        if self.desired_agl is not None:
            z, valid = self._ground([xy])
            if valid[0]:
                height = float(z[0]+self.desired_agl)
            elif self.ground_fallback == 'hold_current':
                height = float(getattr(self, 'latest_position', self.home)[2])
                self.ground_fallback_points += 1
            else:
                raise ValueError('NO_OBSERVED_GROUND_SUPPORT')
        return dict(target_world=[float(xy[0]), float(xy[1]), height],
            role=role, corridor_id=corridor_id)

    def _endpoint_safety(self, snapshot, position, yaw):
        # One activation rebuilt the same obstacle tree up to three times from the
        # same snapshot and pose (~0.04 s on the control thread, run c49e9da0).
        # Callers copy before changing the frame, so one cached instance is safe.
        key = (float(snapshot['timestamp']), tuple(np.asarray(position, float).tolist()), float(yaw))
        cached = getattr(self, '_safety_cache', None)
        if cached is None or cached[0] is not snapshot or cached[1] != key:
            from .subgoal_inputs import _safety
            safety = _safety(snapshot, position, yaw, self.flight, self.operating_bounds,
                self.height_bounds).with_clearance_reserve(self.planning_clearance_reserve)
            self._safety_cache = cached = (snapshot, key, safety)
        return cached[2]

    @staticmethod
    def _endpoint_clear(safety, target):
        # The BEV crop is a presentation bound, not an occupancy restriction on
        # reverse flight. Reuse the exact obstacle tree, body and world bounds.
        endpoint = copy(safety)
        endpoint.frame_position = np.asarray(target)
        return endpoint.clear(np.asarray(target)[None])

    def _wait_for_native_geometry(self, now, selected):
        self.blocked_goal = dict(target_world=selected['target_world'],
            candidate_id=selected.get('candidate_id', self.route_candidate_id), reason='observed_endpoint_occupied',
            actual_position=self.latest_position.tolist(), time=float(now))
        self.route, self.commitment, self.continuation = [], None, None
        self.route_request_id, self.route_candidate_id = None, None
        self.requires_handoff_stop, self.recovery_pending = True, True
        self.handoff_reason = 'observed_endpoint_occupied'
        self._reject('CANDIDATE_NO_LONGER_OBSERVED', now, self.blocked_goal)

    def _lift_guidance(self, start, target, after=None):
        points = [np.asarray(start, float), np.asarray(target, float)]
        if after is not None:
            points.append(np.asarray(after, float))
        points = np.asarray(points)
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
        if self.desired_agl is None or len(points) < 2:
            return points
        dense = np.concatenate([np.linspace(a, b,
            max(2, int(np.ceil(np.linalg.norm(b[:2]-a[:2])/.5))+1))[:-1]
            for a, b in zip(points[:-1], points[1:])]+[points[-1:]])
        z, valid = self._ground(dense[:, :2])
        if not valid.all() and self.ground_fallback != 'hold_current':
            raise ValueError('NO_OBSERVED_GROUND_ALONG_GUIDANCE')
        ground_height = z+self.desired_agl
        if not valid.all():
            # Unobserved ground: hold the actual start height there.
            ground_height[~valid] = points[0][2]
            self.ground_fallback_points += int((~valid).sum())
        # Smoothly preserve the actual start and fixed goal height without
        # changing the selected XY route or inventing unsupported ground.
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1))]
        if self.max_climb_slope is None:
            residual = dense[:, 2]-ground_height
            blend = np.exp(-distance/.75)
            dense[:, 2] = ground_height+residual*blend
        else:
            # The roll-180 MID-360 sees only up to +7 deg: space above the body
            # near the vehicle is never observed, so a steep nearby climb cannot
            # be certified (run 401c7d8c). Follow ground+AGL but change height by
            # at most max_climb_slope per metre; the end meets the fixed goal
            # height at the same slope, the start is the actual height.
            step, height = np.diff(distance)*self.max_climb_slope, ground_height.copy()
            height[-1] = points[-1][2]
            for i in range(len(height)-2, -1, -1):
                height[i] = np.clip(height[i], height[i+1]-step[i], height[i+1]+step[i])
            height[0] = points[0][2]
            for i in range(1, len(height)):
                height[i] = np.clip(height[i], height[i-1]-step[i-1], height[i-1]+step[i-1])
            dense[:, 2] = height
        dense[0] = points[0]
        dense[-1] = points[-1]
        return dense

    def guidance_points(self, actual_position):
        after = self.continuation if self.current_goal.completion_mode == 'through' else None
        return self._lift_guidance(actual_position, self.current_goal.target_world, after)

    def _history_route(self, corridor_id=None):
        import heapq
        points = self.breadcrumbs+[self.latest_position.tolist()]
        # Build an observed-traversal graph, merging spatial revisits. Reversing
        # the entire mission history would replay every coverage and detour loop.
        # Edges describe historical intentions only; fresh lower checks remain.
        nodes, edges, previous = [], {}, None
        for p in map(np.asarray, points):
            distances = np.linalg.norm(np.asarray(nodes)-p, axis=1) if nodes else []
            index = int(np.argmin(distances)) if len(distances) and min(distances) <= .6 else len(nodes)
            if index == len(nodes):
                nodes.append(p)
                edges[index] = {}
            if previous is not None and previous != index:
                cost = float(np.linalg.norm(nodes[previous]-nodes[index]))
                edges[previous][index] = edges[index][previous] = cost
            previous = index
        start = previous
        if corridor_id is None:
            destinations = {0}
        else:
            corridor = next(c for c in self.structure.snapshot()['corridors'] if c['id'] == corridor_id)
            destinations = {i for i, p in enumerate(nodes) if
                abs(p[:2]@corridor['normal']-corridor['offset_m']) < .6 and
                corridor['span'][0] <= p[:2]@corridor['direction'] <= corridor['span'][1]}
        queue, best, parent, end = [(0., start)], {start:0.}, {}, None
        while queue:
            cost, index = heapq.heappop(queue)
            if cost != best[index]:
                continue
            if index in destinations:
                end = index
                break
            for neighbor, distance in edges[index].items():
                value = cost+distance
                if value < best.get(neighbor, np.inf):
                    best[neighbor], parent[neighbor] = value, index
                    heapq.heappush(queue, (value, neighbor))
        if end is None:
            return []
        path = [end]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        raw = [self.latest_position]+[nodes[i] for i in path[1:]]
        simplified, anchor, i = [], raw[0], 1
        while i < len(raw):
            end = i
            while end+1 < len(raw):
                candidate = raw[end+1]
                chord = candidate-anchor
                length = np.linalg.norm(chord)
                if length > 4.5 or length < .3:
                    break
                middle = np.asarray(raw[i:end+1])
                fraction = np.clip((middle-anchor)@chord/(length*length), 0., 1.)
                if np.max(np.linalg.norm(middle-(anchor+fraction[:, None]*chord), axis=1)) > .12:
                    break
                end += 1
            simplified.append(raw[end])
            anchor, i = raw[end], end+1
        route = [dict(target_world=p.tolist(), role='RETURN' if corridor_id is None else 'REJOIN',
            corridor_id=corridor_id) for p in simplified]
        if not route and np.linalg.norm(self.latest_position-nodes[end]) > .3:
            route = [dict(target_world=nodes[end].tolist(), role='RETURN' if corridor_id is None else 'REJOIN',
                corridor_id=corridor_id)]
        if corridor_id is None and route:
            route[-1]['target_world'] = self.home.tolist()
        return route

    def _plan(self, position):
        structure = self.structure.snapshot()
        corridors = structure.get('corridors', [])
        self.structure_ambiguous = bool(structure.get('identity_ambiguous'))
        if self.route:
            return
        if self.decision_policy == 'qwen' and (self.recovery_pending or self.commitment is not None):
            self._plan_native(position, structure)
            return
        if self.intent in ('RETURN_HOME', 'REVISIT_CORRIDOR'):
            if self.intent == 'RETURN_HOME' and np.linalg.norm(position-self.home) <= .3:
                if self.resume_after:
                    self._restore_context(self.resume_context)
                else:
                    self.intent = 'FINISH'
                    if self.coverage_result == 'IN_PROGRESS':
                        self.coverage_result = 'ABORTED'
                return
            target_corridor = next((c for c in corridors if c['id'] == self.preferred_corridor), None)
            at_revisit = bool(target_corridor and self.active_corridor == self.preferred_corridor and
                abs(position[:2]@target_corridor['normal']-target_corridor['offset_m']) < .8 and
                target_corridor['span'][0]-.25 <= position[:2]@target_corridor['direction'] <= target_corridor['span'][1]+.25)
            if self.intent == 'REVISIT_CORRIDOR' and at_revisit:
                self._event('revisit_completed', self.last_time, corridor_id=self.preferred_corridor,
                    command_id=self.active_command, actual_position=position.tolist())
                self.temporary_corridor = self.preferred_corridor if self.resume_after else None
                self.intent = 'COVER' if self.resume_after else 'PAUSE'
                return
            self.route = self._history_route(self.preferred_corridor if self.intent == 'REVISIT_CORRIDOR' else None)
            if not self.route:
                self.intent = 'PAUSE'
            return
        if self.intent == 'VISIT_CORRIDOR' and self.decision_policy == 'qwen':
            self._plan_native(position, structure)
            return
        if self.intent != 'COVER':
            return
        if self.decision_policy == 'qwen':
            self._plan_native(position, structure)
            return
        if not corridors:
            count = self.probes.get('bootstrap', 0)
            if count >= 6:
                self.coverage_result, self.intent = 'UNRESOLVED', 'RETURN_HOME'
                self.resume_after = False
                self._event('structure_bootstrap_budget', self.last_time)
            else:
                self.probes['bootstrap'] = count+1
                d = np.array([np.cos(self.initial_yaw), np.sin(self.initial_yaw)])
                self.route = [self._anchor(position[:2]+1.5*d, 'BOOTSTRAP')]
            return
        if structure.get('identity_ambiguous'):
            self.coverage_result, self.intent = 'UNRESOLVED', 'RETURN_HOME'
            self.resume_after = False
            return
        remaining = [c for c in corridors if self.ledger.gaps(c)]
        temporary = next((c for c in corridors if c['id'] == self.temporary_corridor), None)
        if temporary is not None and not self.ledger.gaps(temporary):
            context = self.resume_context or dict(intent='COVER', corridor_id=None)
            completed = self.temporary_corridor
            self._restore_context(context)
            self._event('interrupted_task_restored', self.last_time,
                restored_intent=self.intent, restored_corridor=self.preferred_corridor,
                completed_temporary_corridor=completed)
            return
        if remaining:
            chosen = min(remaining, key=lambda c:(c['id'] != (self.temporary_corridor or self.preferred_corridor or self.active_corridor),
                abs(position[:2]@c['normal']-c['offset_m'])))
            d, n = np.asarray(chosen['direction']), np.asarray(chosen['normal'])
            current = next((c for c in corridors if c['id'] == self.active_corridor), None)
            if current and chosen['id'] != current['id'] and abs(position[:2]@n-chosen['offset_m']) > 1.:
                # Connect only around a confirmed row end, never through a row.
                side = 'low' if abs(position[:2]@d-current['span'][0]) < abs(position[:2]@d-current['span'][1]) else 'high'
                end = current['ends'][side]
                if not end['confirmed']:
                    self._probe_end(current, side, position)
                    return
                edge = min(current['ends']['low']['s'], chosen['ends']['low']['s'])-2.2 if side == 'low' else max(current['ends']['high']['s'], chosen['ends']['high']['s'])+2.2
                self.route = [self._anchor(d*edge+n*current['offset_m'], 'TRANSIT'),
                    self._anchor(d*edge+n*chosen['offset_m'], 'TRANSIT', chosen['id'])]
                return
            fixed_d = np.asarray(self.ledger.directions[chosen['id']])
            scale, bias = float(d@fixed_d), float(chosen['offset_m']*(n@fixed_d))
            if scale < .9:
                self.coverage_result, self.intent, self.resume_after = 'UNRESOLVED', 'RETURN_HOME', False
                return
            gaps = [[(a-bias)/scale, (b-bias)/scale] for a, b in self.ledger.gaps(chosen)]
            s = float(position[:2]@d)
            a, b = min(gaps, key=lambda g:min(abs(s-g[0]), abs(s-g[1])))
            if s < a-.25:
                target = a
            elif s > b+.25:
                target = b
            else:
                target = b if abs(s-a) < abs(s-b) else a
            target = s+np.clip(target-s, -4., 4.)
            if abs(target-s) < .35:
                # Cross a short interior gap with overlap, rather than issue a
                # zero-length target or erase that gap using goal tolerance.
                target = min(chosen['span'][1], b+.5) if abs(s-a) < abs(s-b) else max(chosen['span'][0], a-.5)
            self.route = [self._anchor(d*target+n*chosen['offset_m'], 'INSPECT', chosen['id'])]
            self.preferred_corridor = None
            return
        evidence = self._completion(structure)
        if evidence['eligible']:
            self.coverage_result, self.intent = 'COMPLETE_UNDER_ASSUMPTIONS', 'RETURN_HOME'
            self.resume_after = False
            self.automatic_return = True
            self._event('coverage_completion_candidate', self.last_time, evidence=evidence)
            return
        # Actively verify unresolved endpoints before considering termination.
        for c in sorted(corridors, key=lambda c:c['id'] != self.active_corridor):
            for side in ('low', 'high'):
                if not c['ends'][side]['confirmed']:
                    self._probe_end(c, side, position)
                    return
        for side in ('low', 'high'):
            if not structure.get('outer_sides', {}).get(side, {}).get('confirmed'):
                self._probe_outer(structure, side, position)
                return

    def _plan_native(self, position, structure):
        from .coverage_candidates import coverage_choices, commitment_limit
        self.choices = []
        self.rule_reference = None
        corridors = structure.get('corridors', [])
        safety = self._endpoint_safety(self.latest, position, self.latest_yaw) if self.latest is not None else None

        def anchor(xy, role, corridor_id):
            point = self._anchor(xy, role, corridor_id)
            if safety is not None and not self._endpoint_clear(safety, point['target_world']):
                raise ValueError('OBSERVED_ENDPOINT_OCCUPIED')
            return point
        self.structure_ambiguous = bool(structure.get('identity_ambiguous'))
        if self.structure_ambiguous:
            if self.commitment is not None:
                self._event('commitment_dropped', self.last_time, reason='STRUCTURE_IDENTITY_AMBIGUOUS',
                    commitment=self.commitment, actual_position=position.tolist())
            self.commitment = None
            self.coverage_result = 'UNRESOLVED'
        else:
            if self.coverage_result == 'UNRESOLVED':
                self.coverage_result = 'IN_PROGRESS'
        if not structure.get('identity_ambiguous') and self.commitment is not None:
            c = next((c for c in corridors if c['id'] == self.commitment['corridor_id']), None)
            if c is not None:
                d, n = np.asarray(c['direction']), np.asarray(c['normal'])
                sign = self.commitment['direction']
                s = float(position[:2]@d)
                outer = self.commitment.get('kind') == 'outer_observation'
                finished = bool(outer and structure.get('outer_sides', {}).get(
                    self.commitment['side'], {}).get('confirmed'))
                offset = self.commitment.get('offset_m', c['offset_m'])
                try:
                    limit = commitment_limit(self.commitment, c)
                except ValueError:
                    limit = np.nan
                if not finished and (limit is None or np.isfinite(limit)):
                    for distance in (4., 3., 2., 1.):
                        target = float(np.clip(s+sign*distance, *c['span']))
                        if limit is not None:
                            target = min(target, limit) if sign > 0 else max(target, limit)
                        if sign*(target-s) <= .35:
                            continue
                        try:
                            point = self._anchor(d*target+n*offset,
                                'OBSERVE' if outer else 'INSPECT', None if outer else c['id'])
                            self._lift_guidance(position, point['target_world'])
                            if safety is None or not self._endpoint_clear(safety, point['target_world']):
                                continue
                            self.route = [point]
                            return
                        except ValueError:
                            pass
            self._event('coverage_intent_boundary', self.last_time,
                commitment=self.commitment, actual_position=position.tolist())
            ended = next((side for side, end in c['ends'].items()
                if abs(float(position[:2]@c['direction'])-end['s']) <= .8), None) if c else None
            self.last_completed_intention = dict(deepcopy(self.commitment),
                actual_end=ended, end_confirmed=bool(c and ended and c['ends'][ended].get('confirmed')),
                actual_position=position.tolist(), time=float(self.last_time), task_revision=self.task_revision)
            self.commitment = None
        if self.intent == 'VISIT_CORRIDOR':
            target = self._visit_target(corridors)
            self.task_spec['target_corridor'] = target
            if target is not None and self.active_corridor == target and not self.ledger.gaps(
                    next(c for c in corridors if c['id'] == target)):
                self._event('visit_completed', self.last_time, corridor_id=target,
                    lateral_index=self._lateral(corridors).get(target, {}).get('lateral_index'),
                    command_id=self.active_command, actual_position=position.tolist())
                if self.resume_after:
                    self._restore_context(self.resume_context or dict(intent='COVER'))
                    self._event('interrupted_task_restored', self.last_time,
                        restored_intent=self.intent, restored_corridor=self.preferred_corridor,
                        completed_temporary_corridor=target, command_id=self.active_command)
                else:
                    self.intent = 'PAUSE'
                self.choices = []
                return
        evidence = self._completion(structure)
        if evidence['eligible'] and self.coverage_result != 'COMPLETE_UNDER_ASSUMPTIONS':
            self.coverage_result = 'COMPLETE_UNDER_ASSUMPTIONS'
            self._event('coverage_completion_candidate', self.last_time, evidence=evidence)
        if evidence['eligible'] and np.linalg.norm(position-self.home) <= .3:
            self.intent = 'FINISH'
            return
        if evidence['eligible'] and self.intent == 'COVER':
            # Coverage is complete: the remaining task is the return. Aisle
            # changes earn nothing (run 832f49c6 re-entered a finished aisle).
            self.choices = self._headland_return(position, corridors, structure, anchor)
        elif not structure.get('identity_ambiguous'):
            self.choices = coverage_choices(structure, position, self.active_corridor,
                self.initial_yaw, anchor, self.probes, observation_yaw=self.latest_yaw,
                bootstrap_budget=self.max_bootstrap_observations)
            current = next((c for c in corridors if c['id'] == self.active_corridor), None)
            if current is not None and self.ledger.covered.get(current['id']):
                d, n = np.asarray(current['direction']), np.asarray(current['normal'])
                fixed = np.asarray(self.ledger.directions[current['id']])
                scale, bias = float(d@fixed), float(current['offset_m']*(n@fixed))
                s = float(position[:2]@d)
                for index, (a, b) in enumerate(self.ledger.gaps(current)):
                    if scale < .9:
                        break
                    a, b = (a-bias)/scale, (b-bias)/scale
                    sign = 1 if s < (a+b)/2 else -1
                    end = np.clip(b+.2 if sign > 0 else a-.2, *current['span'])
                    side = 'high' if sign > 0 else 'low'
                    open_ended = current['id']+('_positive' if sign > 0 else '_negative')
                    if (abs(end-current['span'][1 if sign > 0 else 0]) < .05
                            and not current['ends'][side].get('confirmed')
                            and any(c['candidate_id'] == open_ended for c in self.choices)):
                        # Up to an unconfirmed span end, a repair only duplicates the
                        # open-ended inspection and then stops while the aisle is
                        # still being discovered (run 41af0d9a: two mid-aisle stops).
                        continue
                    point = None
                    for distance in (4., 3., 2., 1.):
                        target = min(s+distance, end) if sign > 0 else max(s-distance, end)
                        if sign*(target-s) <= .35:
                            continue
                        try:
                            point = anchor(d*target+n*current['offset_m'], 'INSPECT', current['id'])
                            break
                        except ValueError:
                            pass
                    if point is None:
                        continue
                    identifier = current['id']+f'_gap_repair_{index}'
                    endpoint = d*end+n*current['offset_m']
                    self.choices.append(dict(point, candidate_id=identifier, id=identifier,
                        route=[point], commitment=dict(corridor_id=current['id'], direction=sign,
                            until_s=float(end), scope=dict(kind='bounded_gap_repair',
                                frame_direction=d.tolist(), endpoint_world_xy=endpoint.tolist(),
                                endpoint_s=float(endpoint@d)))))
            if current is not None and self.ledger.covered.get(current['id']) and not self.ledger.gaps(current):
                # A finished aisle offers no new coverage in either direction. Keep
                # re-inspection only when nothing else can make progress.
                again = {current['id']+'_positive', current['id']+'_negative'}
                progress = [c for c in self.choices if c['candidate_id'] not in again
                            and not c['candidate_id'].startswith('scan_')]
                if progress:
                    self.choices = [c for c in self.choices if c['candidate_id'] not in again]
            if current is not None:
                reference = self._spec_choices(current, corridors)
                if self.decision_input == 'point_choice':
                    self.rule_reference = [c['candidate_id'] for c in reference]
                else:
                    self.choices = reference
        if np.linalg.norm(position-self.home) > .3:
            route = self._history_route()
            if route:
                self.choices.append(dict(route[0], candidate_id='RETURN_HOME', id='RETURN_HOME',
                    label='RETURN_HOME', kind='observed_coverage_choice', route=route,
                    requested_intent='RETURN_HOME', commitment=None, probe_key=None))

    def _headland_return(self, position, corridors, structure, anchor):
        """Return outside every row end instead of replaying the S history.

        From the end of the last aisle the flown history leads back through every
        aisle (~420 m on train/006); the headland beyond all row ends is short and
        was observed from each aisle end. Lower checks still gate every segment.
        """
        if not corridors:
            return []
        d = np.asarray(corridors[0]['direction'], float)
        n = np.array([-d[1], d[0]])
        if any(abs(d@np.asarray(c['direction'])) < .98 for c in corridors):
            return []
        s = float(position[:2]@d)
        lows = [c['ends']['low']['s'] for c in corridors]
        highs = [c['ends']['high']['s'] for c in corridors]
        side = 'low' if abs(s-min(lows)) <= abs(s-max(highs)) else 'high'
        edge = min(lows)-2.2 if side == 'low' else max(highs)+2.2
        home = np.asarray(self.home, float)
        waypoints = [d*edge+n*float(position[:2]@n), d*edge+n*float(home[:2]@n), home[:2]]
        area = (structure.get('public_work_area') or {}).get('xy_bounds')
        dense = [np.asarray(position[:2], float)]
        for target in waypoints:
            count = max(1, int(np.ceil(np.linalg.norm(target-dense[-1])/4.)))
            dense += [dense[-1]+(target-dense[-1])*k/count for k in range(1, count+1)]
        route = []
        try:
            for xy in dense[1:]:
                if area is not None and not (area[0]-.5 <= xy[0] <= area[1]+.5 and area[2]-.5 <= xy[1] <= area[3]+.5):
                    return []
                route.append(dict(anchor(xy, 'RETURN', None), role='RETURN'))
        except ValueError:
            return []
        route = [point for point in route if np.linalg.norm(np.asarray(point['target_world'])-position) > .3]
        if not route:
            return []
        route[-1]['target_world'] = home.tolist()
        return [dict(route[0], candidate_id='RETURN_HOME_HEADLAND', id='RETURN_HOME_HEADLAND',
            label='RETURN_HOME_HEADLAND', kind='observed_coverage_choice', route=route,
            requested_intent='RETURN_HOME', commitment=None, probe_key=None, headland_side=side)]

    def _spec_choices(self, current, corridors):
        """Remove cross-row options that the bound task specification forbids.

        COVER+S: finish the current aisle, prefer an uncovered neighbour, else
        transit toward the nearest open aisle (run fd5c0103 skipped aisle 1 and
        shuttled between two finished aisles for 1745 s). COVER+free keeps the
        preference for uncovered aisles but may leave an unfinished one.
        VISIT_CORRIDOR may leave any aisle and only moves toward its target.
        A subset restricts which aisles count as open. Qwen chooses among the rest.
        """
        choices = self.choices
        order = self._lateral(corridors)

        def target(c):
            # Outer-flank checks are boundary evidence, not aisle changes.
            commitment = c.get('commitment') or {}
            if not c.get('connection_from_corridor') or commitment.get('kind') == 'outer_observation':
                return None
            return commitment.get('corridor_id')
        crossing = [c for c in choices if target(c) is not None]
        if not crossing:
            return choices
        here = order.get(current['id'], {}).get('lateral_index')

        def toward(goal):
            # A headland transfer straight onto the goal aisle beats re-entering
            # the finished aisles in between.
            closer = [c for c in crossing if target(c) in order
                      and abs(order[target(c)]['lateral_index']-goal) < abs(here-goal)]
            direct = [c for c in closer if order[target(c)]['lateral_index'] == goal]
            return direct or closer
        if self.intent == 'VISIT_CORRIDOR':
            goal = order.get(self.task_spec.get('target_corridor'), {}).get(
                'lateral_index', (self.task_spec.get('target') or {}).get('lateral_index'))
            if goal is None or here is None:
                return choices
            if goal == here:
                return [c for c in choices if target(c) is None]
            closer = toward(goal)
            # Leaving an unfinished aisle is allowed: the instruction asked for it.
            return ([c for c in choices if target(c) is None and c['role'] != 'INSPECT'] + closer
                    if closer else choices)
        subset = self._spec_corridor_ids(corridors)
        open_ids = {c['id'] for c in corridors if self.ledger.gaps(c) and (subset is None or c['id'] in subset)}
        inspect_here = [c for c in choices if not target(c) and (c.get('commitment') or {}).get('corridor_id')
                        == current['id'] and c['role'] == 'INSPECT']
        if self.task_spec.get('order', 'S') == 'S' and current['id'] in open_ids and inspect_here:
            # Finish the current aisle (or repair it) before leaving it.
            return [c for c in choices if target(c) is None]
        useful = [c for c in crossing if target(c) in open_ids]
        if self.task_spec.get('order', 'S') == 'S' and not subset:
            # S order moves to an adjacent aisle; skipping one would break the S.
            adjacent = [c for c in useful if here is not None and target(c) in order
                        and abs(order[target(c)]['lateral_index']-here) == 1]
            useful = adjacent
        if useful:
            return [c for c in choices if target(c) is None or c in useful]
        # Only finished neighbours: transit toward the nearest open aisle.
        open_index = [order[i]['lateral_index'] for i in open_ids if i in order]
        if here is None or not open_index:
            return choices
        closer = toward(min(open_index, key=lambda i:abs(i-here)))
        return [c for c in choices if target(c) is None or c in closer] if closer else choices

    def _native_goal(self, decision, selected, request_id, now, position, velocity, yaw,
                     checked_snapshot, *, continuation=False):
        """Bind a model-selected intention or extend that exact intention."""
        target = np.asarray(selected['target_world'], float)
        if np.linalg.norm(target-position) < .3:
            self._reject('ALREADY_AT_TARGET', now)
            return None
        safety = self._endpoint_safety(checked_snapshot, position, yaw)
        if not self._endpoint_clear(safety, target):
            if continuation:
                self._wait_for_native_geometry(now, selected)
            else:
                self._reject('CANDIDATE_NO_LONGER_OBSERVED', now, 'observed endpoint occupied')
            return None
        if selected.get('connection_from_corridor') and not self._validate(
                checked_snapshot, position, yaw, selected, safety=safety)[0]:
            self._reject('CANDIDATE_NO_LONGER_OBSERVED', now, 'connection end no longer reachable')
            return None
        after, direction = None, None
        commitment = selected.get('commitment', self.commitment)
        if selected['role'] == 'INSPECT' and commitment:
            c = next((c for c in self.structure.snapshot().get('corridors', [])
                if c['id'] == commitment['corridor_id']), None)
            if c is not None:
                from .coverage_candidates import commitment_limit
                d, n = np.asarray(c['direction']), np.asarray(c['normal'])
                sign, s = commitment['direction'], float(target[:2]@d)
                following = s+sign*4.
                try:
                    limit = commitment_limit(commitment, c)
                except ValueError:
                    limit = np.nan
                if (c['span'][0]+.5 < following < c['span'][1]-.5
                        and abs(position[:2]@n-c['offset_m']) <= .8
                        and (limit is None or sign*(following-limit) < 0.)):
                    try:
                        after = self._anchor(d*following+n*c['offset_m'], 'INSPECT', c['id'])['target_world']
                        self._lift_guidance(position, target, after)
                        if self._endpoint_clear(safety, after):
                            direction = tuple(np.r_[sign*d, 0.])
                        else:
                            after = None
                    except ValueError:
                        after = None
        elif selected['role'] in ('TRANSIT', 'RETURN', 'REJOIN'):
            # Pass interior route points without a measured stop (run f95290de
            # stopped every 4 m on its return: 67 handoff stops). Sharp corners
            # and the final point keep the stop.
            route = self.route if continuation else (selected.get('route') or [selected])
            if len(route) > 1:
                following = np.asarray(route[1]['target_world'], float)
                arrive, leave = (target-position)[:2], (following-target)[:2]
                if (np.linalg.norm(arrive) > .3 and np.linalg.norm(leave) > .3
                        and arrive@leave >= np.cos(np.deg2rad(60))*np.linalg.norm(arrive)*np.linalg.norm(leave)
                        and self._endpoint_clear(safety, following)):
                    after = following
                    direction = tuple(np.r_[arrive/np.linalg.norm(arrive), 0.])
        try:
            self._lift_guidance(position, target, after)
        except ValueError as error:
            if continuation:
                self._fail(str(error), now)
            else:
                self._reject('GROUND_SUPPORT', now, str(error))
            return None
        revision = self.current_goal.revision+1
        delta = target-position
        goal = Subgoal(f'coverage_{revision:03d}_{decision["candidate_id"]}', revision, tuple(target),
            arrival_yaw=None if after is not None else float(np.arctan2(delta[1], delta[0])),
            position_tolerance=.2, speed_tolerance=.05, dwell_seconds=0. if after is not None else .5,
            timeout_seconds=self.goal_timeout, no_motion_seconds=self.no_motion_seconds,
            no_progress_seconds=self.no_progress_seconds,
            completion_mode='through' if after is not None else 'stop',
            approach_direction=direction, lateral_tolerance=.8 if after is not None else None,
            height_tolerance=.5 if after is not None else None)
        self.goals.append(goal)
        self.index += 1
        self.monitor = GoalMonitor(goal, now)
        self._goal_feedback = self.monitor.update(now, position, velocity, yaw)
        self._selected = dict(decision, role=selected['role'], corridor_id=selected.get('corridor_id'))
        self.continuation = after
        self.hold_started, self.next_policy_time = None, -np.inf
        self.requires_handoff_stop = False
        self.status, self.reason, self.pending = 'RUNNING', '', None
        self.handoff_reason = 'goal_reached'
        event = 'coverage_continuation_activated' if continuation else 'goal_activated'
        details = dict(decision=decision, target_world=target.tolist(),
            actual_position=position.tolist(), actual_velocity=velocity.tolist(),
            revalidated_observation_time=float(checked_snapshot['timestamp']),
            corridor_id=selected.get('corridor_id'), commitment=commitment,
            role=selected['role'], completion_mode=goal.completion_mode)
        if continuation:
            details.update(source_request_id=request_id, parent_candidate_id=self.route_candidate_id)
        else:
            details.update(request_id=request_id)
        self._event(event, now, **details)
        return goal

    def _probe_outer(self, structure, side, position):
        key = 'outer:'+side
        count = self.probes.get(key, 0)
        if count >= 3:
            self.coverage_result, self.intent = 'UNRESOLVED', 'RETURN_HOME'
            self.resume_after = False
            self._event('outer_probe_budget', self.last_time, side=side)
            return
        self.probes[key] = count+1
        rows = sorted(structure['rows'], key=lambda r:r['offset_m'])
        row = rows[0] if side == 'low' else rows[-1]
        d, n = np.asarray(row['direction']), np.asarray(row['normal'])
        spans = np.concatenate([np.asarray(r['spans']).reshape(-1, 2) for r in rows])@d
        near_low = abs(position[:2]@d-spans.min()) < abs(position[:2]@d-spans.max())
        end = float(spans.min()-2.2 if near_low else spans.max()+2.2)
        sign = -1 if side == 'low' else 1
        offset = row['offset_m']+sign*(3.5+count)
        # Go around the observed outer end before probing the stand's flank.
        self.route = [self._anchor(d*end+n*(position[:2]@n), 'TRANSIT'),
            self._anchor(d*end+n*offset, 'TRANSIT'),
            self._anchor(d*np.mean(spans)+n*offset, 'OBSERVE')]

    def _probe_end(self, corridor, side, position):
        key = corridor['id']+':'+side
        count = self.probes.get(key, 0)
        if count >= 3:
            self.coverage_result, self.intent = 'UNRESOLVED', 'RETURN_HOME'
            self.resume_after = False
            self._event('end_probe_budget', self.last_time, key=key)
            return
        self.probes[key] = count+1
        end = corridor['ends'][side]
        d, n = np.asarray(corridor['direction']), np.asarray(corridor['normal'])
        sign = -1 if side == 'low' else 1
        xy = d*(end['s']+sign*(2.5+count))+n*corridor['offset_m']
        if np.linalg.norm(xy-position[:2]) < .35:
            xy += sign*d
        self.route = [self._anchor(xy, 'OBSERVE', corridor['id'])]

    def advance(self, now, position, velocity, yaw, reference_settled=True):
        if self.status != 'WAITING_NEXT' or not reference_settled:
            return None
        if self.requires_handoff_stop and np.linalg.norm(velocity) > .05:
            return None
        if self.decision_policy == 'qwen' and now < self.next_policy_time and self.pending is None:
            return None
        if self.decision_policy == 'qwen' and self.preparation is not None:
            if not self.preparation.done():
                return None
            try:
                prepared = self.preparation.result()
            except Exception:
                prepared = {}
            if prepared.get('observation_only'):
                self.preparation, self.pending = None, None
                self._event('observation_input_saved', now, request_id=prepared['payload']['request_id'],
                    reason=prepared['payload']['admission_reason'])
                self._reject('GROUND_SUPPORT', now, prepared['payload']['admission_reason'])
                return None
        if not self._command_tick(now) or self.intent == 'PAUSE':
            return None
        if (self.intent == 'FINISH' and self.pending is None and self.preparation is None
                and np.linalg.norm(velocity) <= .05 and self.allows_stop()):
            # The bound task ends with a measured stop at home; once coverage is
            # complete and the vehicle is at rest there, stopping is required,
            # not a choice (run 678862f7: Qwen answered HOLD+OBSERVE ten times).
            self.stop_requested = True
            self.status, self.reason = 'COMPLETE', 'policy_finished'
            self._event('policy_stop_requested', now, actual_position=np.asarray(position).tolist(),
                actual_speed_mps=float(np.linalg.norm(velocity)), source='task_spec_terminal_stop')
            return None
        if self.structure.last_error:
            self._fail('STRUCTURE:'+self.structure.last_error, now)
            return None
        if self.pending is None and self.preparation is None:
            from .spline_runtime import observation_age
            if (self.decision_policy == 'qwen'
                    and (self.requires_handoff_stop or self.current_goal.completion_mode != 'through')
                    and (self.latest is None or observation_age(now, time.perf_counter(),
                        self.latest['timestamp'], self.latest['captured']) > self.max_observation_age)):
                # At a stop every path below needs a fresh observation. Planning
                # anyway on each 10 ms tick rebuilt the endpoint map (0.026 s per
                # tick) and slowed the observation worker until every result was
                # stale (run 8ba92099: no recovery request for 150 s). A through
                # gate still decides at once (it may have to brake).
                return None
            self._plan(np.asarray(position))
            while self.route and self.route[0]['role'] in ('TRANSIT', 'RETURN', 'REJOIN') and np.linalg.norm(
                    np.asarray(self.route[0]['target_world'])-position) < .3:
                self.route.pop(0)
            if (self.decision_policy == 'qwen' and self.route and self.route_request_id
                    and (self.current_goal.completion_mode == 'through' or np.linalg.norm(velocity) <= .05)):
                if (self.latest is None or observation_age(now, time.perf_counter(),
                        self.latest['timestamp'], self.latest['captured']) > self.max_observation_age):
                    return None
                selected = dict(self.route[0], candidate_id=f'continue_{self.current_goal.revision+1}')
                decision = dict(candidate_id=selected['candidate_id'], behavior={
                    'INSPECT':'FOLLOW', 'TRANSIT':'CHANGE_AISLE', 'OBSERVE':'FOLLOW',
                    'RETURN':'RETURN', 'REJOIN':'REJOIN'}[selected['role']])
                valid, snapshot, _ = self._validate(self.latest, np.asarray(position), yaw, selected)
                if not valid:
                    if not self._endpoint_clear(self._endpoint_safety(snapshot, position, yaw), selected['target_world']):
                        self._wait_for_native_geometry(now, selected)
                        return None
                    if selected.get('confirmed_ends') and self.current_goal.completion_mode == 'stop':
                        self.route, self.commitment, self.choices = [], None, []
                        self.route_request_id, self.route_candidate_id = None, None
                        self._event('connection_reopened', now, reason='confirmed_end_premise_lost')
                        return None
                    self._fail('COMMITTED_INTENT_GROUND_OR_BOUNDS', now)
                    return None
                return self._native_goal(decision, selected, self.route_request_id, now,
                    np.asarray(position), np.asarray(velocity), yaw, snapshot, continuation=True)
            if (self.decision_policy == 'qwen' and not self.requires_handoff_stop
                    and self.current_goal.completion_mode == 'through'
                    and not self.route):
                if self.structure_ambiguous:
                    # A structure flicker dropped the commitment (run 41af0d9a:
                    # a far aisle vanished for 5 s mid-aisle). Brake at the gate
                    # and decide again at rest once the structure settles.
                    self.requires_handoff_stop = True
                    self.handoff_reason = 'structure_identity_ambiguous'
                    self._event('continuation_suspended', now, reason='STRUCTURE_IDENTITY_AMBIGUOUS',
                        actual_position=np.asarray(position).tolist())
                    return None
                self._fail('COMMITTED_INTENT_CONTINUATION_UNAVAILABLE', now)
                return None
            if (self.decision_policy != 'qwen' and self.intent not in ('FINISH', 'PAUSE')
                    and not self.route and not self.choices):
                return None
        return super().advance(now, position, velocity, yaw, reference_settled)

    def _prepare(self, snapshot, position, yaw, request, folder, history, last_rejection):
        if self.decision_policy == 'qwen':
            return self._prepare_native(snapshot, position, yaw, request, folder, last_rejection, history)
        from .subgoal_inputs import render_candidate_bev
        if self.intent == 'FINISH':
            c = dict(candidate_id='HOLD', id='HOLD', label='任务结束并保持停稳',
                target_world=position.tolist(), relative_body_xyz_m=[0., 0., 0.], role='STOP', kind='hold')
        else:
            c = deepcopy(self.route[0])
            c.update(candidate_id=f'anchor_{self.current_goal.revision+1}',
                id=f'anchor_{self.current_goal.revision+1}', label=c['role'],
                kind='observed_task_anchor',
                relative_body_xyz_m=world_to_local([c['target_world']], position, yaw)[0].tolist())
        candidates = [c]
        # A hold alternative is retained; repeated HOLD does not earn progress.
        if c['candidate_id'] != 'HOLD':
            candidates.append(dict(candidate_id='HOLD', id='HOLD', label='继续观察，不能提前结束',
                target_world=position.tolist(), relative_body_xyz_m=[0., 0., 0.], role='HOLD', kind='hold'))
        image = folder/'input.png'
        render_candidate_bev(snapshot, position, yaw, self.home, candidates, image, self.flight,
            operating_bounds=self.operating_bounds, height_bounds=self.height_bounds,
            trail=self.breadcrumbs, anchor_yaw=self.initial_yaw)
        state = dict(command=self.command_text, intent=self.intent, task_revision=self.task_revision,
            corridor=self.active_corridor, last_departed_corridor=self.departed_corridor,
            coverage=self.ledger.snapshot(self.structure.snapshot().get('corridors', [])),
            candidates=candidates, last_rejection=last_rejection)
        prompt = ('根据指令、在线覆盖账本和候选选择下一步，只输出candidate_id与behavior。'
            '覆盖调度器只提供当前任务的在线子目标，没有真值路线。'
            'INSPECT和BOOTSTRAP使用FOLLOW；TRANSIT使用CHANGE_AISLE；OBSERVE观察位置使用FOLLOW；'
            'REJOIN使用REJOIN；RETURN使用RETURN。只有intent=FINISH才能HOLD+REQUEST_STOP。'
            '其余HOLD使用OBSERVE，不能宣称巡检完成。'+json.dumps(state, ensure_ascii=False))
        payload = dict(request_id=request, task_revision=self.task_revision, prompt=prompt,
            candidates=candidates, source_time=float(snapshot['timestamp']), input_provenance='online scans, measured ledger, command memory only')
        _write(folder/'input.json', payload)
        return dict(payload=payload, image=str(image), prompt=prompt, candidates=candidates)

    def _prepare_native(self, snapshot, position, yaw, request, folder, last_rejection, history=()):
        import hashlib
        from .coverage_view import render_coverage_input
        if self.intent == 'FINISH':
            candidates = []
        elif self.route:
            candidates = [dict(self.route[0], route=deepcopy(self.route),
                candidate_id='RETURN_PATH' if self.intent == 'RETURN_HOME' else 'REJOIN_PATH',
                requested_intent=self.intent, commitment=None)]
        else:
            candidates = deepcopy(self.choices)
        safety = self._endpoint_safety(snapshot, position, yaw) if candidates else None
        admitted, rejected, slots = [], [], set()
        for c in sorted(candidates, key=lambda c:-c.get('step_distance', 0.)):
            slot = c.get('observation_slot')
            if slot is not None and slot in slots:
                continue
            checks = {}
            if self._validate(snapshot, position, yaw, c, safety=safety, checks=checks)[0]:
                admitted.append(c)
                if slot is not None:
                    slots.add(slot)
            else:
                rejected.append(dict(candidate_id=c['candidate_id'], checks=checks))
        candidates = admitted
        for c in candidates:
            c.update(id=c['candidate_id'], label=c.get('label', c['candidate_id']),
                kind='observed_coverage_choice',
                relative_body_xyz_m=world_to_local([c['target_world']], position, yaw)[0].tolist())
        # Candidate ordering carries no route recommendation. The complete
        # geometry-generated set and permutation are saved for independent audit.
        candidates.sort(key=lambda c:hashlib.sha256((request+c['candidate_id']).encode()).digest())
        structure = self.structure.snapshot()
        completion = self._completion(structure)
        return_allowed = self.intent != 'COVER' or completion['eligible']
        dropped = {}
        if self.decision_input == 'point_choice':
            from .coverage_candidates import candidate_intent_metrics
            from .coverage_prompt import select_candidates
            # A visit may go to a finished aisle; coverage tasks skip ones without gain.
            gains = ({c['candidate_id']: candidate_intent_metrics(c, position, structure, self.ledger)['new_coverage_m']
                      for c in candidates} if self.intent == 'COVER' else None)
            candidates, dropped = select_candidates(candidates, return_allowed, position, gains=gains)
        candidates.append(dict(candidate_id='HOLD', id='HOLD', label='HOLD',
            target_world=position.tolist(), relative_body_xyz_m=[0., 0., 0.],
            role='STOP' if self.intent == 'FINISH' else 'HOLD', kind='hold'))
        image = folder/'input.png'
        if self.decision_input != 'point_choice':
            render_coverage_input(snapshot, position, yaw, self.home, candidates, image, self.flight,
                self.operating_bounds, self.height_bounds, self.breadcrumbs, self.initial_yaw,
                structure, self.ledger)
        order = self._lateral(structure.get('corridors', []))
        rows = sorted(structure.get('rows', []), key=lambda r: r['offset_m'])
        outer = structure.get('outer_sides', {})

        def outer_open(c):
            # An outermost aisle may have another beyond it until that side is
            # confirmed; saying so afterwards kept Qwen hovering instead of
            # returning (runs 4b639dae, c65a882d).
            return not rows or any(r['id'] in c.get('bordering_rows', []) and not outer.get(side, {}).get('confirmed')
                                   for side, r in (('low', rows[0]), ('high', rows[-1])))
        topology = [dict(corridor_id=c['id'], direction=c['direction'], normal=c['normal'],
            offset_m=round(c['offset_m'], 2), discovered_span=np.round(c['span'], 2).tolist(),
            ends={side:bool(c['ends'][side]['confirmed']) for side in ('low', 'high')},
            outer_open=outer_open(c), **order.get(c['id'], {}))
            for c in structure.get('corridors', [])]
        from .coverage_candidates import observed_task_memory, candidate_intent_metrics
        memory, geometry_features = observed_task_memory(structure, position, yaw, history, candidates)
        coverage = self.ledger.snapshot(structure.get('corridors', []))
        uncovered = {row['corridor_id']:row['uncovered_length_m'] for row in coverage['corridors']}
        current = next((c for c in structure.get('corridors', []) if c['id'] == self.active_corridor), None)
        at_end = next((side for side, end in current['ends'].items() if end.get('confirmed') and
            abs(float(position[:2]@current['direction'])-end['s']) <= .8), None) if current else None
        near_open = next((side for side, end in current['ends'].items() if not end.get('confirmed') and
            abs(float(position[:2]@current['direction'])-end['s']) <= 8.), None) if current else None
        public = []
        for c in candidates:
            commitment = c.get('commitment') or {}
            metrics = candidate_intent_metrics(c, position, structure, self.ledger)
            target_order = order.get(commitment.get('corridor_id', c.get('corridor_id')), {})
            public.append(dict(candidate_id=c['candidate_id'], role=c['role'],
                allowed_behavior={'INSPECT':'FOLLOW', 'BOOTSTRAP':'FOLLOW', 'OBSERVE':'FOLLOW',
                    'TRANSIT':'CHANGE_AISLE', 'RETURN':'RETURN', 'REJOIN':'REJOIN',
                    'HOLD':'OBSERVE', 'STOP':'REQUEST_STOP'}[c['role']],
                corridor_id=commitment.get('corridor_id', c.get('corridor_id')),
                inspection_direction=commitment.get('direction'),
                target_world=np.round(c['target_world'], 2).tolist(),
                relative_body_xyz_m=np.round(c['relative_body_xyz_m'], 2).tolist(),
                observation_slot=c.get('observation_slot'), probe=c.get('probe_key'),
                frontier_id=c.get('frontier_id'), geometry=geometry_features[c['candidate_id']],
                connection_end=c.get('connection_end'), connection_distance_m=c.get('connection_distance_m'),
                reposition_distance_m=round(float(np.linalg.norm(np.asarray(c['target_world'])-position)), 2),
                **metrics,
                corridor_uncovered_m=uncovered.get(commitment.get('corridor_id', c.get('corridor_id'))),
                corridor_lateral_index=target_order.get('lateral_index'),
                corridor_observed_edge=target_order.get('observed_edge'),
                skipped_lateral_indices=[order.get(cid, {}).get('lateral_index')
                    for cid in c.get('skipped_corridors', [])] or None,
                skipped_uncovered_m=round(sum(uncovered.get(cid) or 0. for cid in c.get('skipped_corridors', [])), 2)
                    if c.get('skipped_corridors') else None,
                commitment_scope={key:value for key, value in commitment.items()
                    if key not in ('corridor_id', 'direction', 'source_request_id', 'candidate_id')} or None,
                route_scope=[[point['role'], point.get('corridor_id')]
                    for point in c.get('route', [])]))
            if c.get('requested_intent') == 'RETURN_HOME':
                public[-1].update(authorized_after_coverage=completion['eligible'],
                    authorized_now=return_allowed)
        state = dict(command=self.command_text, intent=self.intent, task_revision=self.task_revision,
            resume_after=self.resume_after,
            task_spec=dict(self.task_spec, target_lateral_index=order.get(
                self.task_spec.get('target_corridor'), {}).get('lateral_index')),
            position_world=np.round(position, 2).tolist(), current_corridor=self.active_corridor,
            current_corridor_lateral_index=order.get(self.active_corridor, {}).get('lateral_index'),
            observed_corridor_count=len(order),
            yaw_world_rad=round(float(yaw), 3), initial_yaw_world_rad=self.initial_yaw,
            initial_yaw_source='public operator initial pose, not an estimated row direction',
            current_corridor_uncovered_m=uncovered.get(self.active_corridor), at_confirmed_end=at_end, current_end=at_end,
            unconfirmed_near_end=near_open,
            previous_corridor=self.departed_corridor, previous_intention=self.last_completed_intention,
            current_intention=self.commitment, recovery_pending=self.recovery_pending,
            blocked_goal=self.blocked_goal, recoveries_used=self.recoveries,
            recovery_budget=self.max_recoveries,
            observation_memory=memory,
            public_work_area=structure.get('public_work_area'),
            unresolved_row_gaps=structure.get('unresolved_row_gaps', []),
            topology=topology, coverage=coverage,
            completion=completion, coverage_result=self.coverage_result,
            return_authorized_now=return_allowed,
            candidates=public, last_rejection=last_rejection)
        if self.decision_input == 'point_choice':
            return self._point_choice_input(snapshot, position, yaw, request, folder, structure, state,
                candidates, public, order, coverage, rejected, dropped)
        from .coverage_candidates import compact_prompt_state, discovery_phase_context
        phase, phase_context = discovery_phase_context(state)
        state['phase'] = phase
        prompt_state = compact_prompt_state(state)
        prompt = ('按指令完成全林S形巡检，自主选择通道和方向。在线候选随机排列，无推荐。'
            '沿当前通道同向巡至确认行末，再经行末连接相邻未巡通道反向巡检；补齐作业区内全部uncovered区间。'
            '只统计public_work_area内的通道；作业区外不计覆盖，也不需要补查。'
            '无当前通道可entry_observe接近；entry_via_low/high经该通道已确认行末外侧进入，不横穿树行。受阻可scan绕障观察。blocked_goal执行无进展或超时，需要新观测恢复。'
            '通道ID编号不代表空间顺序：lateral_index为已观测通道按横向位置的序号，相差1才是相邻通道；'
            'observed_edge标记当前已观测的最外侧通道(low/high)，外侧可能仍有未发现通道。'
            'S形从一侧最外通道开始逐条相邻推进，不跳过中间通道，避免留下只能非相邻补飞的外侧通道。'
            'gap_repair补巡；connection_observe仅查连接。unresolved_row_gaps是未知区域，不是通道；frontier/scan仅补观测。'
            'observed_span仅为已发现支撑；ends=false须继续观测，probe补查端点或边界；无新通道不代表完成。'
            '左图BEV是当前高度切片：灰色未知、橙色阻挡、浅色已观测；右图圆圈为树干候选，线为拟合树行，空白未知。'
            '三维高度和避障由下层实时检查，候选不是全程自由证明。eligible=true后返航，到家且FINISH才结束。'
            '行为：INSPECT/OBSERVE/BOOTSTRAP用FOLLOW，TRANSIT用CHANGE_AISLE，'
            'RETURN用RETURN，REJOIN用REJOIN；HOLD用OBSERVE，只有FINISH时用REQUEST_STOP。'
            'candidate/corridor/scope的rows按对应columns读取，全部候选保留；所有_mm为毫米，_m为米。'
            'corridor_uncovered_m是整廊剩余，不是该方向收益；new_coverage_mm/repeated_coverage_mm是该承诺内已知可补巡/重复长度。'
            'inspection_range_s_mm按该廊coverage.along_direction，起点到承诺终点有序。'
            'explicit_route_distance_mm是显式3D路段长；total_route_distance_mm加上同向XY延伸至当前观测边界或固定补巡端点，未计未知延伸和绕障。'
            'boundary_observation是该动作可补查的外侧及缺测摘要，不能预先保证补证成功。route_scope为[角色,通道]完整阶段。'
            'route_scope_runs为[角色,通道,重复次数]，按序展开。'
            'scope_ref引用固定补巡世界端点平面，到达即结束承诺；candidate_defaults为全部候选的共同字段。'
            '只输出JSON，恰好含candidate_id和behavior，不输出解释或坐标。\n'
            +json.dumps(prompt_state, ensure_ascii=False, separators=(',', ':')))
        if self.intent == 'COVER' and not return_allowed:
            prompt += ('\n当前阶段仍是COVER，覆盖证据不完整：返回起点当前未获授权。'
                '用户要求全林巡检完成后才返航。根据本次观测选择巡检、补观测或恢复候选；'
                '暂时不能推进可选择HOLD和OBSERVE，不能把未完成任务改成返航结束。')
        elif self.intent == 'RETURN_HOME':
            prompt += '\n当前RETURN_HOME意图已获授权，继续选择返航或必要的观察恢复。'
        if phase_context:
            prompt += '\n'+phase_context
        payload = dict(request_id=request, task_revision=self.task_revision, prompt=prompt,
            source_time=float(snapshot['timestamp']), candidates=candidates,
            state=state, prompt_state=prompt_state, prompt_phase_context=phase_context,
            rejected_geometry=rejected,
            prompt_projection=dict(schema='compact_coverage_state_v2', numeric_decimals=3,
                candidates='all candidates in candidate_columns/candidate_rows',
                coverage='all required_mm, covered_mm, uncovered_mm intervals retained in integer millimetres',
                scopes='scope_ref links complete shared scope_columns/scope_rows; dot paths name nested fields'),
            movement_alternatives=sum(c['role'] not in ('HOLD', 'STOP') and
                (c.get('requested_intent') != 'RETURN_HOME' or return_allowed) for c in candidates),
            input_provenance='online scans, observed topology and ground, measured coverage; no truth route')
        observation_only = self.intent != 'FINISH' and payload['movement_alternatives'] == 0
        payload.update(model_requested=not observation_only,
            admission_reason='NO_SUPPORTED_MOVEMENT' if observation_only else 'OBSERVED_CANDIDATES_AVAILABLE')
        _write(folder/'input.json', payload)
        return dict(payload=payload, image=str(image), prompt=prompt, candidates=candidates,
            observation_only=observation_only)

    def _point_choice_input(self, snapshot, position, yaw, request, folder, structure, state,
                            candidates, public, order, coverage, rejected, dropped):
        """point_choice_v1: two world-aligned images, short text, one letter back."""
        from .coverage_images import render_bev, render_global
        from .coverage_prompt import PROTOCOL, assign_labels, candidate_orders, point_choice_prompt
        labels = assign_labels(candidates)
        images = [folder/'global.png', folder/'bev.png']
        view = dict(global_image=render_global(structure, coverage, order, position, yaw, self.home,
                self.breadcrumbs, candidates, labels, images[0]),
            bev_image=render_bev(self._endpoint_safety(snapshot, position, yaw), position, yaw,
                candidates, labels, images[1], ground=self._ground))
        metrics = {p['candidate_id']: p for p in public}
        prompt = point_choice_prompt(state, candidates, labels, metrics, facts=self.decision_facts)
        # The BEV is still drawn for the record when only the global image is sent.
        sent = images if self.decision_images == 'both' else images[:1]
        view.update(sent_images=[path.name for path in sent], prompt_facts=self.decision_facts)
        orders = [dict(image=[str(path) for path in sent], prompt=prompt, labels=labels)]
        for k, ordered in enumerate(candidate_orders(candidates, self.decision_orders)[1:], 1):
            # The same decision in a rotated order: letters, images and text redrawn.
            letters = assign_labels(ordered)
            paths = [folder/f'global_o{k}.png']+([folder/f'bev_o{k}.png'] if self.decision_images == 'both' else [])
            render_global(structure, coverage, order, position, yaw, self.home, self.breadcrumbs,
                ordered, letters, paths[0])
            if len(paths) > 1:
                render_bev(self._endpoint_safety(snapshot, position, yaw), position, yaw,
                    ordered, letters, paths[1], ground=self._ground)
            orders.append(dict(image=[str(path) for path in paths], labels=letters,
                prompt=point_choice_prompt(state, ordered, letters, metrics, facts=self.decision_facts)))
        # Everything needed to redraw and re-ask this decision offline.
        overview = structure.get('overview')
        np.savez_compressed(folder/'replay.npz', points=np.asarray(snapshot['points'], float).reshape(-1, 3),
            evidence=np.asarray(snapshot['evidence'], float).reshape(-1, 3),
            trail=np.asarray(self.breadcrumbs, float).reshape(-1, 3),
            overview=np.zeros((0, 0), np.uint8) if overview is None else np.asarray(overview['flags']))
        (folder/'structure.json').write_text(json.dumps(dict(
            {k: structure.get(k) for k in ('timestamp', 'rows', 'corridors', 'stem_candidates', 'outer_sides',
                'unresolved_row_gaps', 'public_work_area', 'merged_rows', 'identity_ambiguous')},
            overview=None if overview is None else dict(origin=overview['origin'], resolution=overview['resolution']),
            terrain_truth=self.terrain_truth is not None), ensure_ascii=False))
        payload = dict(request_id=request, protocol=PROTOCOL, task_revision=self.task_revision, prompt=prompt,
            source_time=float(snapshot['timestamp']), candidates=candidates, labels=labels,
            rule_reference=self.rule_reference, dropped_candidates=dropped, rejected_geometry=rejected,
            state=state, view=view, movement_alternatives=sum(c['role'] not in ('HOLD', 'STOP') for c in candidates),
            input_provenance='online scans, observed topology and ground, measured coverage; no truth route')
        if len(orders) > 1:
            payload['orders'] = [dict(labels=o['labels'], images=[Path(p).name for p in o['image']],
                prompt=o['prompt']) for o in orders]
        observation_only = self.intent != 'FINISH' and payload['movement_alternatives'] == 0
        payload.update(model_requested=not observation_only,
            admission_reason='NO_SUPPORTED_MOVEMENT' if observation_only else 'OBSERVED_CANDIDATES_AVAILABLE')
        _write(folder/'input.json', payload)
        return dict(payload=payload, image=[str(path) for path in sent], prompt=prompt, candidates=candidates,
            labels=labels, response_schema='choice', observation_only=observation_only,
            **(dict(orders=orders) if len(orders) > 1 else {}))

    def _validate(self, snapshot, position, yaw, selected, *, safety=None, checks=None):
        checks = {} if checks is None else checks
        p = np.asarray(selected['target_world'])
        xmin, xmax, ymin, ymax = self.operating_bounds
        # Semantic intent only: the lower executor checks every current segment.
        valid = bool(np.isfinite(p).all() and xmin < p[0] < xmax and ymin < p[1] < ymax
            and self.height_bounds[0] <= p[2] <= self.height_bounds[1])
        checks['within_operating_bounds'] = valid
        if valid and selected.get('role') not in ('HOLD', 'STOP'):
            safety = safety or self._endpoint_safety(snapshot, position, yaw)
            checks['endpoint_clear'] = self._endpoint_clear(safety, p)
            valid = checks['endpoint_clear']
        if valid and self.desired_agl is not None:
            try:
                self._lift_guidance(position, p)
                checks['terrain_supported'] = True
            except ValueError:
                checks['terrain_supported'] = False
                valid = False
        if valid and selected.get('confirmed_ends'):
            corridors = {c['id']:c for c in self.structure.snapshot().get('corridors', [])}
            valid = all(corridors.get(identifier, {}).get('ends', {}).get(side, {}).get('confirmed')
                for identifier, side in selected['confirmed_ends'])
        if valid and selected.get('connection_from_corridor'):
            corridor = next((c for c in self.structure.snapshot().get('corridors', [])
                if c['id'] == selected['connection_from_corridor']), None)
            from .coverage_candidates import distance_before_end
            side = selected['connection_end']
            checks['near_connection_end'] = bool(corridor and distance_before_end(
                float(position[:2]@corridor['direction']), corridor['ends'][side]['s'], side) <= 8.)
            valid = checks['near_connection_end']
        if valid and selected.get('requires_observed_connection'):
            from .subgoal_inputs import _safety
            safety = safety or _safety(snapshot, position, yaw, self.flight, self.operating_bounds, self.height_bounds)
            points = self._lift_guidance(position, p)
            dense = np.concatenate([np.linspace(a, b,
                max(2, int(np.ceil(np.linalg.norm(b-a)/.05))+1)) for a, b in zip(points[:-1], points[1:])])
            checks.update(obstacle_clear=bool(safety.clear(dense)),
                body_observed=bool(safety.known(dense, position, body_padding=1e-5)))
            valid = checks['obstacle_clear'] and checks['body_observed']
        return valid, snapshot, position

    def _activate(self, decision, selected, pending, now, position, velocity, yaw, checked_snapshot):
        if pending['task_revision'] != self.task_revision:
            self._reject('SUPERSEDED', now)
            return None
        if self.intent == 'FINISH':
            self._reject('FINISH_IS_NOT_MOVEMENT', now)
            return None
        if self.decision_policy == 'qwen':
            if selected.get('requested_intent') == 'RETURN_HOME' and self.intent == 'COVER':
                evidence = self._completion(self.structure.snapshot())
                if not evidence['eligible']:
                    self._reject('EARLY_RETURN_UNFINISHED', now, dict(unresolved=evidence['unresolved']))
                    return None
            expected = {'INSPECT':'FOLLOW', 'BOOTSTRAP':'FOLLOW', 'OBSERVE':'FOLLOW',
                'TRANSIT':'CHANGE_AISLE', 'RETURN':'RETURN', 'REJOIN':'REJOIN'}[selected['role']]
            if decision['behavior'] != expected:
                self._reject('CHOICE_BEHAVIOR_MISMATCH', now)
                return None
            goal = self._native_goal(decision, selected, pending['request_id'], now,
                position, velocity, yaw, checked_snapshot)
            if goal is not None:
                self.route = deepcopy(selected.get('route', [selected]))
                for anchor in self.route:
                    for key in ('confirmed_ends', 'requires_observed_connection',
                            'connection_from_corridor', 'connection_end'):
                        if key in selected:
                            anchor[key] = deepcopy(selected[key])
                self.commitment = deepcopy(selected.get('commitment'))
                self.route_request_id, self.route_candidate_id = pending['request_id'], selected['candidate_id']
                if self.commitment:
                    self.commitment.update(source_request_id=pending['request_id'],
                        candidate_id=selected['candidate_id'])
                    self.recovery_pending = False
                if selected.get('probe_key'):
                    key = selected['probe_key']
                    self.probes[key] = self.probes.get(key, 0)+1
                if selected.get('requested_intent'):
                    self.intent = selected['requested_intent']
                    self.automatic_return = self.coverage_result == 'COMPLETE_UNDER_ASSUMPTIONS'
                    self.recovery_pending = False
                self.choices = []
            return goal
        goal = super()._activate(decision, selected, pending, now, position, velocity, yaw, checked_snapshot)
        if goal is not None:
            self._selected.update(role=selected['role'], corridor_id=selected.get('corridor_id'))
            self.handoff_reason = 'goal_reached'
        return goal

    def allows_stop(self):
        return bool(self.intent == 'FINISH' and (self.coverage_result != 'COMPLETE_UNDER_ASSUMPTIONS'
            or self._completion(self.structure.snapshot())['eligible']))

    def _fail(self, reason, now):
        if (self.decision_policy == 'qwen' and self._monitor_recovery_allowed
                and reason in ('no_motion', 'no_progress', 'timeout')):
            if self.recoveries >= self.max_recoveries:
                return super()._fail('local_recovery_budget:'+reason, now)
            self.recoveries += 1
            self._completed_since_recovery = 0
            self.blocked_goal = dict(goal_id=self.current_goal.goal_id,
                target_world=list(self.current_goal.target_world), reason=reason,
                candidate_id=(self._selected or {}).get('candidate_id'),
                actual_position=self.latest_position.tolist(), time=float(now))
            self.task_revision += 1
            self.route, self.choices, self.commitment, self.continuation = [], [], None, None
            self.route_request_id, self.route_candidate_id = None, None
            self.pending = None
            self.requires_handoff_stop, self.recovery_pending = True, True
            self.hold_started, self.next_policy_time = None, -np.inf
            self.status, self.reason = 'WAITING_NEXT', 'local_execution_blocked'
            self.handoff_reason = 'local_execution_blocked:'+reason
            self._event('coverage_recovery_requested', now, blocked_goal=self.blocked_goal,
                recoveries_used=self.recoveries, recovery_budget=self.max_recoveries)
            return
        return super()._fail(reason, now)

    def _reject(self, reason, now, details=None):
        if self.decision_policy == 'qwen' and reason in (
                'OBSERVE_RETRY', 'CANDIDATE_NO_LONGER_OBSERVED', 'GROUND_SUPPORT'):
            self.pending = None
            self.hold_started = now if self.hold_started is None else self.hold_started
            self.next_policy_time = now+2.
            self._event('observation_hold', now, reason=reason, details=details,
                waiting_seconds=float(now-self.hold_started))
            if now-self.hold_started >= self.no_motion_seconds:
                self._fail('observation_no_progress', now)
            return
        return super()._reject(reason, now, details)

    def _model_error(self, response, now):
        if str(response.get('error', '')).startswith('MODEL_OUTPUT:'):
            self._reject('MODEL_OUTPUT_SCHEMA', now, response['error'])
        else:
            super()._model_error(response, now)

    def feedback(self):
        result = super().feedback()
        if hasattr(self, 'ledger'):
            result.update(intent=self.intent, corridor_id=self.active_corridor,
                coverage_result=self.coverage_result, task_revision=self.task_revision,
                recoveries_used=self.recoveries, recovery_budget=self.max_recoveries,
                requires_handoff_stop=self.requires_handoff_stop, blocked_goal=self.blocked_goal)
        return result

    def definition(self):
        structure = self.structure.snapshot()
        # The overview raster is image input (saved per decision), not JSON.
        structure.pop('overview', None)
        if 'terrain' in structure:
            # The query index (arrays, triangulation) is runtime-only, not JSON.
            structure['terrain'] = {k:v for k, v in structure['terrain'].items() if not k.startswith('_')}
        return dict(protocol=(('qwen_native_forest_s_coverage_v2' if self.structure.snapshot().get('public_work_area')
                else 'qwen_native_forest_s_coverage_v1') if self.decision_policy == 'qwen'
                else 'observed_online_coverage_pilot_v1'), instruction=self.instruction,
            height_mode='OBSERVED_AGL' if self.desired_agl is not None else 'WORLD_Z',
            desired_agl=self.desired_agl, decision_policy=self.decision_policy,
            planning_clearance_reserve_m=self.planning_clearance_reserve,
            recoveries_used=self.recoveries, recovery_budget=self.max_recoveries,
            max_bootstrap_observations=self.max_bootstrap_observations,
            ground_fallback=self.ground_fallback, ground_fallback_points=self.ground_fallback_points,
            terrain_source='truth' if self.terrain_truth is not None else 'observed',
            start=self.home.tolist(), intent=self.intent,
            structure=structure, coverage=self.ledger.snapshot(self.structure.snapshot().get('corridors', [])),
            completion=self._completion(self.structure.snapshot()),
            coverage_result=self.coverage_result, commands=self.command_results,
            entered_corridors=self.entered_corridors, command_file=str(self.command_file),
            limitations='conditional sampled boundary evidence; no general completeness guarantee or semantic payload inspection')
