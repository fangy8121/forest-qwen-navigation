"""Independent, conservative audit of runtime command execution logs.

This module never supplies navigation answers. Expectations come from the raw
operator command/independent test plan, never from the model's compiled answer.
COMPLETE means all requested log evidence exists; it is not a collision or
full-forest coverage certificate. Missing injections are MISSING_EVIDENCE.
"""
from __future__ import annotations

import math
import re
import numpy as np


COMPLETE, MISSING, FAILED = 'COMPLETE', 'MISSING_EVIDENCE', 'FAILED'


def _time(event):
    value = event.get('time', event.get('timestamp'))
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _vector(event, field):
    try:
        value = np.asarray(event[field], float)
        return value if value.shape == (3,) and np.isfinite(value).all() else None
    except (KeyError, TypeError, ValueError):
        return None


def _speed(event):
    value = event.get('actual_speed_mps')
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return float(value)
    velocity = _vector(event, 'actual_velocity')
    return float(np.linalg.norm(velocity)) if velocity is not None else None


def _key(event):
    return event.get('goal_id'), event.get('revision')


def _status(checks):
    return FAILED if any(c['status'] == FAILED for c in checks) else MISSING if any(
        c['status'] == MISSING for c in checks) else COMPLETE


def _check(name, status, detail, evidence=None):
    return dict(check=name, status=status, detail=detail, evidence=evidence)


def _expected(text, plan):
    """Small declared command grammar; unsupported prose needs test expectations.

    This is deliberately not a second general language model. Negated or
    ambiguous commands outside this grammar remain unresolved rather than being
    scored using the tested model's own interpretation.
    """
    intent = plan.get('expected_intent')
    resume = plan.get('expected_resume_after')
    reference = plan.get('reference')
    if intent is None:
        stripped = text.strip().lower()
        if stripped in ('暂停', '停一下', 'pause', 'stop'):
            intent = 'PAUSE'
        elif stripped in ('继续', '继续巡检', '恢复巡检', '继续任务', 'resume'):
            intent = 'RESUME'
        elif not any(word in text for word in ('不要', '别', '无需')):
            if '通道' in text and any(word in text for word in ('刚才那条', '上一条')):
                intent, reference = 'REVISIT_CORRIDOR', 'LAST_DEPARTED'
                if resume is None:
                    resume = '继续' in text or '接着' in text
            elif any(word in text for word in ('返回起点', '返航', '回起点')) and not (
                    '巡检整个' in text or '巡完' in text):
                intent = 'RETURN_HOME'
                if resume is None:
                    resume = bool(re.search(r'(再|之后|然后).*继续', text))
    if reference is None and intent == 'REVISIT_CORRIDOR' and any(
            word in text for word in ('刚才那条', '上一条')):
        reference = 'LAST_DEPARTED'
    restored = plan.get('expected_restored_intent')
    if restored is None and resume and '继续巡检' in text:
        restored = 'COVER'
    return dict(intent=intent, resume_after=resume,
        corridor_id=plan.get('expected_corridor_id'), reference=reference,
        restored_intent=restored)


def _last_departed(events, received):
    # Ordered entry transitions are independent measured task events. Do not
    # recover the answer from command_bound or the model's memory prompt.
    previous, departed = None, None
    for event in events:
        if event is received:
            break
        if event.get('kind') != 'corridor_entered' or _time(event) is None or _time(event) > _time(received):
            continue
        identifier = event.get('corridor_id')
        if identifier and identifier != previous:
            departed = event.get('previous_corridor') or previous or departed
            previous = identifier
    return departed


def _rest_reused(stop, received, until, lower, max_gap=.35):
    """A prior measured stop is reusable only with continuous logged rest.

    This supports the existing lower layer's latched handoff reference; absence
    of a *new* braking manoeuvre does not itself mean an unsafe switch.
    """
    if _speed(received) is None or _speed(received) > .05:
        return None
    anchor = _vector(stop, 'actual_position')
    if anchor is None:
        return None
    observations = [e for e in lower if e.get('event') == 'observation'
        and _time(e) is not None and _time(stop) < _time(e) <= until and _key(e) == _key(stop)]
    if not observations:
        return None
    times = [_time(stop)]+[_time(e) for e in observations]+[until]
    if max(b-a for a, b in zip(times[:-1], times[1:])) > max_gap:
        return None
    for event in observations:
        p = _vector(event, 'actual_position')
        vectors = [_vector(event, field) for field in (
            'actual_velocity', 'reference_velocity', 'actual_acceleration', 'reference_acceleration')]
        if p is None or any(v is None for v in vectors) or np.linalg.norm(p-anchor) > .05:
            return None
        if any(np.linalg.norm(v) > limit for v, limit in zip(vectors, (.05, .05, .15, .15))):
            return None
    return dict(mode='reused_latched_stop_with_continuous_rest', measured_stop_time=_time(stop),
        rest_observations=len(observations), continuous_until=until)


def _zero_gap_proof(events, identifier, since, until):
    checkpoints = [e for e in events if e.get('kind') == 'coverage_checkpoint'
        and _time(e) is not None and since <= _time(e) <= until]
    for event in reversed(checkpoints):
        rows = event.get('ledger', {}).get('corridors', [])
        row = next((r for r in rows if r.get('corridor_id') == identifier), None)
        if row is None:
            continue
        gaps = row.get('uncovered')
        length = row.get('uncovered_length_m')
        if not isinstance(gaps, list) or not isinstance(length, (int, float)) or not math.isfinite(length):
            return None
        return dict(time=_time(event), corridor_id=identifier, uncovered=gaps,
            uncovered_length_m=float(length), clear=not gaps and abs(length) <= 1e-8,
            scope='logged task-ledger evidence; not an independent trajectory coverage calculation')
    return None


def audit_coverage_commands(mission_events, lower_decisions, command_plan=None, *, policy=None):
    """Audit raw event dictionaries and an optional independent command plan.

    command_plan is a list of command_id/text/min_speed and optional
    expected_intent, expected_corridor_id, expected_resume_after, reference,
    expected_restored_intent, allow_no_movement. ``at`` alone does not prove an
    injection. A planned command without command_received is never a success.
    Policies are rule, qwen_commands (model language + rule anchors), qwen, and teacher
    (rule language + decision-bench teacher on the point_choice candidates).
    Inputs are not modified. All evidence times use saved simulation timestamps.
    """
    events = sorted((dict(e) for e in mission_events), key=lambda e: (_time(e) is None, _time(e) or 0., e.get('sequence', 0)))
    lower = sorted((dict(e) for e in lower_decisions), key=lambda e: (_time(e) is None, _time(e) or 0.))
    received = [e for e in events if e.get('kind') == 'command_received']
    plans = {str(p['command_id']):dict(p) for p in (command_plan or [])}
    identifiers = list(plans)
    for event in received:
        identifier = str(event.get('command_id', ''))
        if identifier not in identifiers:
            identifiers.append(identifier)
    global_checks, rows = [], []
    modes = dict(rule='RULE_BASELINE', qwen_commands='MODEL_COMMANDS_RULE_ANCHORS', qwen='MODEL_COMMANDS_AND_CANDIDATES',
                 teacher='RULE_COMMANDS_TEACHER_CANDIDATES')
    if policy not in modes:
        global_checks.append(_check('policy_provenance', MISSING, 'Declare rule, qwen_commands or qwen; log syntax alone is not model provenance.'))
    if not identifiers:
        global_checks.append(_check('command_exercise', MISSING, 'No command was planned or received.'))

    activations = [e for e in events if e.get('kind') == 'goal_activated']
    goal_versions = {_key(e):e.get('task_revision') for e in activations}
    combined_activations = [(e, 'mission') for e in activations]+[
        (e, 'lower') for e in lower if e.get('event') == 'goal_activated']
    stale = []
    for event, source in combined_activations:
        t = _time(event)
        previous = [r for r in received if _time(r) is not None and t is not None and _time(r) <= t]
        if not previous:
            continue
        latest = previous[-1]
        actual_revision = event.get('task_revision') if source == 'mission' else goal_versions.get(_key(event))
        required_revision = latest.get('task_revision')
        if not isinstance(actual_revision, int) or not isinstance(required_revision, int):
            global_checks.append(_check('activation_version', MISSING, 'Activation or command lacks task revision.',
                dict(source=source, time=t, goal_id=event.get('goal_id'))))
        elif actual_revision < required_revision:
            evidence = dict(source=source, time=t, goal_id=event.get('goal_id'),
                activated_task_revision=actual_revision, required_task_revision=required_revision,
                after_command_id=latest.get('command_id'))
            stale.append(evidence)
            global_checks.append(_check('no_superseded_activation', FAILED, 'An obsolete task activated after a new command.', evidence))
    if received and not stale and not any(c['check'] == 'activation_version' for c in global_checks):
        global_checks.append(_check('no_superseded_activation', COMPLETE,
            'All logged activations respect the latest received command version.', dict(checked=len(combined_activations))))

    for identifier in identifiers:
        checks, plan = [], plans.get(identifier, {})
        completion = None
        matches = [e for e in received if str(e.get('command_id', '')) == identifier]
        row = dict(command_id=identifier, policy_mode=modes.get(policy, 'UNSPECIFIED'), checks=checks)
        if not matches:
            checks.append(_check('received', MISSING, 'Planned command was not observed at runtime.'))
            row['status'] = _status(checks)
            rows.append(row)
            continue
        if len(matches) > 1:
            checks.append(_check('idempotent_reception', FAILED, 'The same command ID was accepted more than once.', dict(count=len(matches))))
        event = matches[0]
        start = _time(event)
        if start is None:
            checks.append(_check('received', MISSING, 'Command lacks finite reception time.'))
            row['status'] = _status(checks)
            rows.append(row)
            continue
        following = next((r for r in received[received.index(event)+1:] if _time(r) is not None), None)
        end = _time(following) if following is not None else math.inf
        text = str(event.get('text', plan.get('text', '')))
        expected = _expected(text, plan)
        row.update(text=text, received_at=start, expected=expected)
        if 'text' in plan and text != plan['text']:
            checks.append(_check('raw_command', FAILED, 'Received text differs from the independent command plan.'))
        speed = _speed(event)
        minimum = float(plan.get('min_speed', 0.))
        checks.append(_check('reception_state', MISSING if speed is None else FAILED if speed+1e-8 < minimum else COMPLETE,
            'Reception speed checked against the independent plan.', dict(actual_speed_mps=speed,
                required_min_speed_mps=minimum, received_while_moving=bool(speed is not None and speed > .05))))
        if expected['intent'] is None:
            checks.append(_check('independent_semantics', MISSING, 'Command is outside the audit grammar; supply independent expected_intent.'))
        window = [e for e in events if _time(e) is not None and start <= _time(e) < end]
        bound = [e for e in window if e.get('kind') == 'command_bound' and e.get('command_id') == identifier]
        bound_command = bound[-1].get('command', {}) if bound else {}
        if expected['intent'] == 'PAUSE' and text.strip().lower() in ('暂停', '停一下', 'pause', 'stop'):
            row['semantic_path'] = 'EXPLICIT_FAST_PAUSE'
            checks.append(_check('command_binding', COMPLETE, 'Exact pause vocabulary uses the explicit fast path.'))
        elif not bound:
            checks.append(_check('command_binding', MISSING, 'No accepted command binding was logged.'))
        else:
            wrong = (expected['intent'] is not None and bound_command.get('intent') != expected['intent']) or (
                expected['resume_after'] is not None and bound_command.get('resume_after') != expected['resume_after'])
            checks.append(_check('command_binding', FAILED if wrong else COMPLETE, 'Compared accepted binding with independent expectations.',
                dict(bound_at=_time(bound[-1]), accepted=bound_command)))
            responses = [e.get('response', {}) for e in window if e.get('kind') == 'command_model_response'
                and _time(e) <= _time(bound[-1])]
            response = responses[-1] if responses else {}
            rule = 'deterministic rule baseline' in str(response.get('source', '')).lower()
            model_evidence = (isinstance(response.get('input_tokens'), int) and response['input_tokens'] > 0
                and isinstance(response.get('raw'), str) and response.get('response_schema') == 'coverage_command' and not rule)
            row['semantic_path'] = 'RULE_BASELINE' if rule else 'MODEL_TOKEN_AND_RAW_EVIDENCE' if model_evidence else 'UNVERIFIED'
            if policy in ('qwen', 'qwen_commands') and not model_evidence:
                checks.append(_check('model_semantics_provenance', MISSING, 'Model policy lacks raw/token command-response evidence.'))

        movement = [e for e in activations if _time(e) is not None and start <= _time(e) < end]
        first_activation = _time(movement[0]) if movement else None
        valid_goal_key = bool(event.get('goal_id') and isinstance(event.get('revision'), int))
        stops = [e for e in lower if valid_goal_key and e.get('event') == 'measured_stop' and _time(e) is not None
            and _key(e) == _key(event) and _time(e) < end and _speed(e) is not None and _speed(e) <= .05]
        direct = next((e for e in stops if _time(e) >= start and (first_activation is None or _time(e) <= first_activation)), None)
        stop_evidence = dict(mode='new_measured_stop', time=_time(direct), goal_id=direct.get('goal_id')) if direct else None
        if stop_evidence is None:
            prior = [e for e in stops if _time(e) < start]
            if prior:
                stop_evidence = _rest_reused(prior[-1], event, first_activation if first_activation is not None else start, lower)
        checks.append(_check('measured_handoff_stop', COMPLETE if stop_evidence else MISSING,
            'Require a matching actual stop or continuously verified reuse of a latched stop.', stop_evidence))
        if movement and not stop_evidence:
            checks.append(_check('activation_after_stop', MISSING, 'Movement activated without sufficient logged stop evidence.'))
        elif movement:
            checks.append(_check('activation_after_stop', COMPLETE, 'First new movement follows verified handoff rest.', dict(time=first_activation)))
        elif expected['intent'] != 'PAUSE' and not plan.get('allow_no_movement', False):
            checks.append(_check('movement_exercised', MISSING, 'No new movement was observed before the next command/end.'))
        if expected['intent'] == 'PAUSE' and movement:
            checks.append(_check('remain_paused', FAILED, 'A new goal activated before a later resume/replacement command.', dict(time=first_activation)))
        if expected['intent'] == 'PAUSE' and stop_evidence:
            completion = dict(kind='measured_handoff_stop',
                time=stop_evidence.get('time', stop_evidence.get('continuous_until', start)),
                source='lower', stop_evidence=stop_evidence)
        elif expected['intent'] == 'RESUME' and movement and stop_evidence:
            # Model binding acknowledges an intent, not actual resumption. A
            # new physical execution goal is the minimum completion evidence.
            completion = dict(kind='goal_activated', time=first_activation,
                source='mission', goal_id=movement[0].get('goal_id'))

        if expected['intent'] == 'REVISIT_CORRIDOR':
            frozen = expected['corridor_id']
            if expected['reference'] == 'LAST_DEPARTED':
                observed_departure = _last_departed(events, event)
                if frozen is not None and frozen != observed_departure:
                    checks.append(_check('independent_reference', FAILED, 'Test expectation conflicts with the measured entry history.'))
                frozen = observed_departure
            row['frozen_target_corridor_id'] = frozen
            checks.append(_check('revisit_identity', MISSING if frozen is None or not bound else
                COMPLETE if bound_command.get('corridor_id') == frozen else FAILED,
                'Bind the target to the reception-time history, never the model answer.',
                dict(expected_corridor_id=frozen, bound_corridor_id=bound_command.get('corridor_id'))))
            revisits = [e for e in window if e.get('kind') == 'revisit_completed' and e.get('command_id') == identifier]
            if not revisits:
                checks.append(_check('revisit_completed', MISSING, 'No actual revisit completion event.'))
            else:
                wrong = frozen is not None and revisits[-1].get('corridor_id') != frozen
                entries = [e for e in window if e.get('kind') == 'corridor_entered'
                    and _time(e) <= _time(revisits[-1])]
                observed_identity = entries[-1].get('corridor_id') if entries else None
                wrong = wrong or bool(observed_identity is not None and frozen is not None and observed_identity != frozen)
                measured = _vector(revisits[-1], 'actual_position') is not None and observed_identity is not None
                checks.append(_check('revisit_completed', FAILED if wrong else COMPLETE if measured else MISSING,
                    'Revisit completion needs the same frozen corridor, entry transition and actual position.',
                    dict(time=_time(revisits[-1]), corridor_id=revisits[-1].get('corridor_id'), observed_entry=observed_identity)))
                if expected['resume_after'] is False:
                    completion = dict(kind='revisit_completed', time=_time(revisits[-1]),
                        source='mission', corridor_id=frozen)
            if expected['resume_after']:
                restored = [e for e in window if e.get('kind') == 'interrupted_task_restored'
                    and e.get('completed_temporary_corridor') == frozen]
                if not restored:
                    checks.append(_check('coverage_and_restoration', MISSING, 'No matching coverage-restoration event.'))
                else:
                    restoration = restored[-1]
                    proof = _zero_gap_proof(events, frozen, start, _time(restoration))
                    expected_restored = expected['restored_intent']
                    if expected_restored is None:
                        checks.append(_check('restored_task_semantics', MISSING,
                            'Supply an independent expected_restored_intent for this resume instruction.'))
                    wrong = bool(expected_restored and restoration.get('restored_intent') != expected_restored)
                    if revisits and _time(restoration) < _time(revisits[-1]):
                        wrong = True
                    checks.append(_check('coverage_and_restoration', FAILED if wrong or proof and not proof['clear'] else
                        COMPLETE if proof else MISSING, 'Require cleared target gaps and restoration after revisit.',
                        dict(restored_at=_time(restoration), restored_intent=restoration.get('restored_intent'), coverage=proof)))
                    completion = dict(kind='interrupted_task_restored', time=_time(restoration),
                        source='mission', corridor_id=frozen, restored_intent=restoration.get('restored_intent'))
        if expected['intent'] == 'VISIT_CORRIDOR':
            # The target is fixed by the independent plan as a lateral index.
            wanted = plan.get('expected_target_lateral_index')
            visits = [e for e in window if e.get('kind') == 'visit_completed' and e.get('command_id') == identifier]
            if wanted is None:
                checks.append(_check('visit_identity', MISSING, 'Supply expected_target_lateral_index for VISIT_CORRIDOR.'))
            elif not visits:
                checks.append(_check('visit_completed', MISSING, 'No actual visit completion event.'))
            else:
                visit = visits[-1]
                entries = [e for e in window if e.get('kind') == 'corridor_entered' and _time(e) <= _time(visit)]
                observed = entries[-1].get('corridor_id') if entries else None
                wrong = visit.get('lateral_index') != wanted or (observed is not None and observed != visit.get('corridor_id'))
                checks.append(_check('visit_completed', FAILED if wrong else COMPLETE if observed else MISSING,
                    'Visit completion needs the planned lateral index, an entry transition and actual position.',
                    dict(time=_time(visit), corridor_id=visit.get('corridor_id'), lateral_index=visit.get('lateral_index'),
                         observed_entry=observed)))
                completion = dict(kind='visit_completed', time=_time(visit), source='mission',
                    corridor_id=visit.get('corridor_id'))
                if expected['resume_after']:
                    restored = [e for e in window if e.get('kind') == 'interrupted_task_restored'
                        and e.get('completed_temporary_corridor') == visit.get('corridor_id')
                        and _time(e) >= _time(visit)]
                    expected_restored = expected['restored_intent'] or 'COVER'
                    checks.append(_check('coverage_and_restoration',
                        MISSING if not restored else COMPLETE if restored[-1].get('restored_intent') == expected_restored else FAILED,
                        'Require restoration of the interrupted task after the visit.',
                        dict(restored_at=_time(restored[-1]) if restored else None)))
                    if restored:
                        completion = dict(kind='interrupted_task_restored', time=_time(restored[-1]),
                            source='mission', corridor_id=visit.get('corridor_id'))
        # Close attribution only when ALL of the command's own evidence is
        # complete. Wrong binding, absent stopping, missing revisit/coverage or
        # model provenance cannot acquire a completion boundary from one marker.
        if completion is not None and _status(checks) != COMPLETE:
            completion = None
        completed_at = completion['time'] if completion is not None else None
        row.update(completed_at=completed_at, completion_evidence=completion,
            failure_attribution_until=completed_at if completed_at is not None else end if math.isfinite(end) else None)
        task_failures = [e for e in window if e.get('kind') == 'task_failed']
        attributable = [e for e in task_failures if completed_at is None or _time(e) <= completed_at]
        subsequent = [e for e in task_failures if completed_at is not None and _time(e) > completed_at]
        if attributable:
            checks.append(_check('task_failure', FAILED,
                'A task failure occurred before command execution/restoration was evidenced.',
                [dict(time=_time(e), reason=e.get('reason')) for e in attributable]))
        row['subsequent_task_failures'] = [dict(time=_time(e), reason=e.get('reason'),
            attribution='after_proven_command_completion') for e in subsequent]
        row['status'] = _status(checks)
        rows.append(row)

    statuses = [c['status'] for c in global_checks]+[r['status'] for r in rows]
    status = FAILED if FAILED in statuses else MISSING if MISSING in statuses or not statuses else COMPLETE
    return dict(schema='coverage_command_audit_v1', status=status, complete=status == COMPLETE,
        policy_mode=modes.get(policy, 'UNSPECIFIED'), commands=rows, global_checks=global_checks,
        mission_task_failures=[dict(time=_time(e), reason=e.get('reason')) for e in events if e.get('kind') == 'task_failed'],
        counts={key:sum(r['status'] == key for r in rows) for key in (COMPLETE, MISSING, FAILED)},
        scope='runtime command semantics, version and measured handoff evidence; not full coverage/safety certification')
