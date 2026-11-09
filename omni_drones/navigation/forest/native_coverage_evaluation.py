"""Post-flight scoring against hidden native forest geometry.

Nothing in this module supplies navigation targets. Keep it outside the runtime
supervisor: the true row count, extents and terrain are evaluation answers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ENDPOINT_TOLERANCE = .25
ROW_CLEARANCE = .75
MAX_SAMPLE_GAP = .5
MAX_SEGMENT_LENGTH = .5
MAX_MEASURED_SPEED = 1.5


def _merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if b-a <= 1e-9:
            continue
        if result and a <= result[-1][1]+1e-7:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([float(a), float(b)])
    return result


def _gaps(low, high, covered):
    result, cursor = [], low
    for a, b in covered:
        if a > cursor+1e-7:
            result.append([float(cursor), float(min(a, high))])
        cursor = max(cursor, b)
        if cursor >= high:
            break
    if cursor < high-1e-7:
        result.append([float(cursor), float(high)])
    return [g for g in result if g[1] > g[0]+1e-7]


def native_evaluation_spec(scene_info):
    """Fit a common row frame from actual stems, preserving unequal row tails."""
    grouped = {}
    for tree in scene_info.get('trees', []):
        if tree.get('row') is not None:
            grouped.setdefault(str(tree['row']), []).append(tree['position'][:2])
    groups = [(identifier, np.asarray(points, float)) for identifier, points in grouped.items()
              if len(points) >= 2]
    if len(groups) < 2:
        raise ValueError('native evaluation requires at least two true tree row groups')
    centered = np.concatenate([p-p.mean(axis=0) for _, p in groups])
    _, vectors = np.linalg.eigh(centered.T@centered)
    d = vectors[:, -1]
    if d[0] < -1e-8 or abs(d[0]) <= 1e-8 and d[1] < 0:
        d = -d
    n = np.array([-d[1], d[0]])
    rows = []
    for identifier, points in groups:
        s, q = points@d, points@n
        if not np.isfinite(points).all() or np.ptp(s) < .5:
            raise ValueError('finite nondegenerate true row geometry required')
        slope, intercept = np.polyfit(s, q, 1)
        rows.append(dict(id=identifier, span=[float(s.min()), float(s.max())],
            slope=float(slope), intercept=float(intercept), offset_m=float(np.median(q))))
    rows.sort(key=lambda r:r['offset_m'])
    corridors = []
    for index, (a, b) in enumerate(zip(rows[:-1], rows[1:])):
        low, high = min(a['span'][0], b['span'][0]), max(a['span'][1], b['span'][1])
        mid = (low+high)/2
        width = (b['slope']-a['slope'])*mid+b['intercept']-a['intercept']
        if width <= 2*ROW_CLEARANCE:
            raise ValueError('true rows do not define a positive inspection strip')
        corridors.append(dict(id=f'truth_{a["id"]}_{b["id"]}', index=index,
            bordering_rows=[a['id'], b['id']], span=[low, high],
            bilateral_span=[max(a['span'][0], b['span'][0]), min(a['span'][1], b['span'][1])],
            low_row=[a['slope'], a['intercept']], high_row=[b['slope'], b['intercept']],
            direction=d.tolist(), normal=n.tolist()))
    return dict(schema='native_forest_route_evaluation_v1', corridors=corridors, rows=rows,
        direction=d.tolist(), normal=n.tolist(), endpoint_tolerance_m=ENDPOINT_TOLERANCE,
        row_centerline_clearance_m=ROW_CLEARANCE,
        denominator='all true adjacent-row union spans, including single-sided unequal tails',
        scope='actual geometric route traversal; not full tree-surface or disease inspection')


def _trace_arrays(trace, actual):
    if trace is None:
        return None, None
    rows = np.asarray(trace, float)
    if rows.ndim != 2 or rows.shape[1] < 7:
        raise ValueError('trace must contain time, actual position and velocity')
    xyz, velocity = (rows[:, 8:11], rows[:, 15:18]) if rows.shape[1] >= 21 else (rows[:, 1:4], rows[:, 4:7])
    if len(rows) == len(actual)-1 and len(rows) and np.allclose(xyz, actual[1:], atol=1e-5):
        dt = np.diff(rows[:, 0])
        dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else .01
        return np.r_[rows[0, 0]-dt, rows[:, 0]], np.vstack([velocity[0], velocity])
    if len(rows) != len(actual) or not np.allclose(xyz, actual, atol=1e-5):
        raise ValueError('trace actual positions must match the evaluated trajectory')
    return rows[:, 0], velocity


def _clip_segments(s, q, corridor, valid):
    """Clip each real segment to the inspection strip and along-span bounds."""
    a, b = np.zeros(len(valid)), np.ones(len(valid))
    lo, hi = corridor['span']
    lm, lb = corridor['low_row']; hm, hb = corridor['high_row']
    values = [s-lo, hi-s, q-lm*s-lb-ROW_CLEARANCE, hm*s+hb-q-ROW_CLEARANCE]
    for value in values:
        v0, delta = value[:-1], np.diff(value)
        parallel = abs(delta) < 1e-12
        valid &= ~(parallel & (v0 < 0))
        root = np.divide(-v0, delta, out=np.zeros_like(delta), where=~parallel)
        a = np.where(delta > 1e-12, np.maximum(a, root), a)
        b = np.where(delta < -1e-12, np.minimum(b, root), b)
    valid &= b > a+1e-10
    start, end = s[:-1]+a*np.diff(s), s[:-1]+b*np.diff(s)
    return _merge(zip(np.minimum(start[valid], end[valid]), np.maximum(start[valid], end[valid])))


def _s_pattern(actual, spec, valid_segments):
    if not len(actual):
        return dict(passed=False, passages=[], first_complete_order=[], reason='empty trajectory')
    s, q = actual[:, :2]@spec['direction'], actual[:, :2]@spec['normal']
    labels = np.full(len(actual), -1, int)
    for c in spec['corridors']:
        lm, lb = c['low_row']; hm, hb = c['high_row']; low, high = c['span']
        inside = ((q-lm*s-lb >= ROW_CLEARANCE) & (hm*s+hb-q >= ROW_CLEARANCE)
            & (s >= low-ENDPOINT_TOLERANCE) & (s <= high+ENDPOINT_TOLERANCE))
        labels[inside] = c['index']
    breaks = np.flatnonzero(np.r_[True, (labels[1:] != labels[:-1]) | ~valid_segments, True])
    passages = []
    for start, end in zip(breaks[:-1], breaks[1:]):
        if labels[start] < 0 or end-start < 2:
            continue
        c, values = spec['corridors'][labels[start]], s[start:end]
        low, high = c['span']
        # Endpoint transitions retain genuine out-and-back traversals within
        # one corridor. Local reversals or gap repairs cannot invent a pass.
        side = np.where(values <= low+ENDPOINT_TOLERANCE+1e-7, -1,
                        np.where(values >= high-ENDPOINT_TOLERANCE-1e-7, 1, 0))
        endpoints = np.flatnonzero(side)
        if len(endpoints) < 2:
            continue
        changes = np.flatnonzero(np.r_[True, np.diff(side[endpoints]) != 0])
        for a, b in zip(changes[:-1], changes[1:]):
            first, last = endpoints[a], endpoints[b]
            passages.append(dict(corridor_id=c['id'], index=c['index'],
                direction=int(side[last]), start_sample=int(start+first), end_sample=int(start+last)))
    first, seen, cutoff = [], set(), len(passages)
    for index, passage in enumerate(passages):
        if passage['index'] not in seen:
            first.append(passage)
            seen.add(passage['index'])
        if len(seen) == len(spec['corridors']):
            cutoff = index+1
            break
    inspection = passages[:cutoff]
    transitions = [(a, b) for a, b in zip(inspection[:-1], inspection[1:]) if a['index'] != b['index']]
    adjacent = all(abs(a['index']-b['index']) == 1 for a, b in transitions)
    alternating = all(a['direction'] == -b['direction'] for a, b in transitions)
    return dict(passed=bool(len(first) == len(spec['corridors']) and adjacent and alternating),
        passages=passages, first_complete_order=[p['corridor_id'] for p in first],
        adjacent_passages=adjacent, alternating_passages=alternating,
        inspection_passage_count=len(inspection), post_coverage_passage_count=len(passages)-cutoff,
        adjacent_first_passes=all(abs(a['index']-b['index']) == 1 for a, b in zip(first[:-1], first[1:])),
        alternating_first_passes=all(a['direction'] == -b['direction'] for a, b in zip(first[:-1], first[1:])),
        invalid_transitions=[dict(previous=a, following=b) for a, b in transitions
            if abs(a['index']-b['index']) != 1 or a['direction'] != -b['direction']],
        scope='chronological adjacent alternating full passes through complete coverage; revisits allowed, first-visit order is diagnostic')


def _true_agl(actual, scene_info):
    protocol = scene_info.get('evaluation_protocol', {})
    bounds = protocol.get('agl_bounds', [1.5, 3.5])
    nominal = float(protocol.get('nominal_agl', 2.5))
    terrain = scene_info.get('evaluation_terrain')
    file = scene_info.get('evaluation_terrain_file')
    if terrain is None and file is not None:
        with np.load(file) as data:
            terrain = {k:data[k] for k in ('xs', 'ys', 'heights')}
    if terrain is None or not len(actual):
        return dict(available=False, reason='true terrain grid was not supplied for post-flight evaluation')
    from forest_generator.terrain import grid_height
    xs, ys, heights = [np.asarray(terrain[k]) for k in ('xs', 'ys', 'heights')]
    if (len(xs) < 2 or len(ys) < 2 or heights.shape != (len(xs), len(ys))
            or not np.isfinite(np.r_[xs, ys, heights.ravel()]).all()):
        return dict(available=False, reason='invalid true terrain grid')
    inside = ((actual[:, 0] >= xs[0]) & (actual[:, 0] <= xs[-1])
        & (actual[:, 1] >= ys[0]) & (actual[:, 1] <= ys[-1]))
    agl = actual[inside, 2]-grid_height(actual[inside, :2], xs, ys, heights)
    violations = int(((agl < bounds[0]-1e-7) | (agl > bounds[1]+1e-7)).sum())
    return dict(available=True, samples=int(len(agl)), outside_terrain_samples=int((~inside).sum()),
        minimum_m=float(agl.min()) if len(agl) else None,
        maximum_m=float(agl.max()) if len(agl) else None,
        nominal_agl_m=nominal, agl_bounds_m=list(bounds), out_of_bounds_samples=violations,
        absolute_error_p95_m=float(np.percentile(abs(agl-nominal), 95)) if len(agl) else None,
        passed=bool(len(agl) == len(actual) and len(agl) and violations == 0),
        scope='true triangle-grid AGL; no extrapolation outside the supplied terrain')


def _work_area_containment(scene_info):
    bounds = scene_info.get('evaluation_protocol', {}).get('work_area_bounds')
    if bounds is None:
        return dict(coverage_scope='UNKNOWN_STAND_BOUNDARY', xy_bounds=None,
            contains_entire_forest=None, reason='no public work area was supplied')
    try:
        b = np.asarray(bounds, float)
        valid = bool(b.shape == (4,) and np.isfinite(b).all() and b[0] < b[1] and b[2] < b[3])
    except (TypeError, ValueError):
        valid = False
    points = np.asarray([t['position'][:2] for t in scene_info.get('trees', [])], float)
    finite_trees = bool(points.ndim == 2 and points.shape[1] == 2 and len(points)
        and np.isfinite(points).all())
    outside = None
    if valid and finite_trees:
        outside = int(((points[:, 0] < b[0]) | (points[:, 0] > b[1])
            | (points[:, 1] < b[2]) | (points[:, 1] > b[3])).sum())
    return dict(coverage_scope='BOUND_PUBLIC_WORK_AREA', xy_bounds=b.tolist() if valid else None,
        valid_bounds=valid, true_tree_count=len(points), outside_tree_centers=outside,
        contains_entire_forest=bool(valid and finite_trees and outside == 0),
        scope='all original true tree centers must be inside; true corridor/tail denominator is unchanged')


def evaluate_native_coverage(actual, final_velocity, home, physical_stop_verified, mission,
                             truth_margin_violation, scene_info, *, trace=None, decisions=None):
    """Evaluate only completed flight data, independently of online row IDs."""
    actual = np.asarray(actual, float).reshape(-1, 3)
    velocity, home = np.asarray(final_velocity, float), np.asarray(home, float)
    if velocity.shape != (3,) or home.shape != (3,) or not np.isfinite(np.r_[actual.ravel(), velocity, home]).all():
        raise ValueError('finite actual XYZ trajectory, home and final velocity required')
    spec = native_evaluation_spec(scene_info)
    times, velocities = _trace_arrays(trace, actual)
    distance = np.linalg.norm(np.diff(actual, axis=0), axis=1)
    valid = distance <= MAX_SEGMENT_LENGTH
    if times is not None:
        dt = np.diff(times)
        valid &= np.isfinite(dt) & (dt > 0) & (dt <= MAX_SAMPLE_GAP)
        valid &= distance <= MAX_MEASURED_SPEED*dt+.02
    s, q = actual[:, :2]@spec['direction'], actual[:, :2]@spec['normal']
    coverage = []
    for corridor in spec['corridors']:
        covered = _clip_segments(s, q, corridor, valid.copy())
        low, high = corridor['span']
        gaps = _gaps(low+ENDPOINT_TOLERANCE, high-ENDPOINT_TOLERANCE, covered)
        length = sum(b-a for a, b in covered)
        coverage.append(dict(truth_corridor_id=corridor['id'], required_span=[low, high],
            bilateral_span=corridor['bilateral_span'], covered_intervals=covered, uncovered_intervals=gaps,
            actual_route_coverage_fraction=float(min(1., length/(high-low))),
            uncovered_length_m=float(sum(b-a for a, b in gaps)), complete=not gaps))
    task = task_specification(mission, spec, home)
    for row, index in zip(coverage, task['truth_lateral_index']):
        row.update(lateral_index_from_home=index, required_by_task=task['required'](index))
    required = [c for c in coverage if c['required_by_task']]
    route_complete = bool(required and all(c['complete'] for c in required))
    pattern = _s_pattern(actual, spec, valid)
    final_home = bool(len(actual) and np.linalg.norm(actual[-1]-home) <= .3)
    quiet_dwell = False
    if times is not None and len(times) and velocities is not None:
        tail = times >= times[-1]-1.-1e-8
        quiet_dwell = bool(np.ptp(times[tail]) >= 1.-.02 and np.isfinite(velocities[tail]).all()
            and np.max(np.linalg.norm(velocities[tail], axis=1)) <= .05
            and np.max(np.linalg.norm(actual[tail]-actual[-1], axis=1)) <= .05
            and np.all(np.diff(times[tail]) > 0) and np.max(np.diff(times[tail])) <= MAX_SAMPLE_GAP)
    stopped = bool(physical_stop_verified and np.linalg.norm(velocity) <= .05 and quiet_dwell)
    state = mission if isinstance(mission, dict) else vars(mission)
    declared = state.get('coverage_result')
    authorized = bool(state.get('status') == 'COMPLETE' and state.get('stop_requested'))
    provenance = audit_native_decisions(decisions, required_corridor_count=len(required),
        corridor_choice_required=task['corridor_choice_required']) if decisions is not None else None
    agl = _true_agl(actual, scene_info)
    work_area = _work_area_containment(scene_info)
    terrain_passed = agl.get('passed', False) if scene_info.get('evaluation_protocol', {}).get('agl_bounds') else None
    # Everything except model provenance: the verdict for the teacher ceiling.
    task_passed = bool(route_complete and (pattern['passed'] or not task['s_pattern_required'])
        and final_home and stopped and authorized
        and truth_margin_violation is False and declared == 'COMPLETE_UNDER_ASSUMPTIONS'
        and terrain_passed is not False and work_area['contains_entire_forest'] is not False)
    passed = bool(task_passed and (provenance is None or provenance['passed']))
    return dict(schema=spec['schema'], truth_route_coverage=coverage,
        actual_full_route_coverage=route_complete, s_pattern=pattern,
        task_specification={k:v for k, v in task.items() if k != 'required'},
        true_corridor_count=len(coverage), final_home=final_home, final_measured_stop=stopped,
        measured_one_second_quiet_dwell=quiet_dwell,
        final_home_error_m=float(np.linalg.norm(actual[-1]-home)) if len(actual) else None,
        truth_margin_clear=truth_margin_violation is False, declared_coverage_result=declared,
        authorized_task_termination=authorized, rejected_discontinuous_segments=int((~valid).sum()),
        false_completion=bool(declared == 'COMPLETE_UNDER_ASSUMPTIONS' and not route_complete),
        full_coverage_passed=passed, task_passed_without_model_provenance=task_passed, terrain_agl=agl, terrain_following_passed=terrain_passed,
        coverage_scope=work_area['coverage_scope'], public_work_area=work_area,
        work_area_contains_entire_forest=work_area['contains_entire_forest'],
        decision_provenance=provenance, evaluation_spec=spec,
        metric_scope=spec['scope'], limitations='route geometry and actual stop; scan quality/tree-surface completeness are separate metrics')


def _inspection_commitment(candidate):
    commitment = candidate.get('commitment') or {}
    cid, direction = commitment.get('corridor_id'), commitment.get('direction')
    if commitment.get('kind') == 'outer_observation' or not cid or direction not in (-1, 1):
        return None
    route = candidate.get('route', [candidate])
    if any(p.get('role') == 'INSPECT' and p.get('corridor_id') == cid for p in route):
        return dict(corridor_id=cid, direction=direction)
    return None


def _same_commitment(expected, actual):
    if not expected:
        return not actual
    keys = ('kind', 'corridor_id', 'direction')
    if expected.get('kind') == 'outer_observation':
        keys += ('side', 'offset_m')
    # Provenance fields may be appended by the executor, but a selected repair
    # boundary must remain exactly the boundary the model was offered.
    return bool(actual and all(expected.get(k) == actual.get(k) for k in keys)
        and all((k in expected) == (k in actual) and expected.get(k) == actual.get(k)
            for k in ('until_s', 'scope')))


def _within_inspection_bound(commitment, target, topology):
    """Check bounded repairs in their immutable model-selection frame."""
    if 'until_s' not in commitment and 'scope' not in commitment:
        return True
    try:
        scope = commitment.get('scope')
        if scope is not None:
            if scope.get('kind') != 'bounded_gap_repair':
                return False
            direction = np.asarray(scope['frame_direction'], float)
            endpoint = np.asarray(scope['endpoint_world_xy'], float)
            bound = float(scope['endpoint_s'])
            if (endpoint.shape != (2,) or not np.isfinite(endpoint).all()
                    or direction.shape != (2,) or abs(endpoint@direction-bound) > 1e-6
                    or ('until_s' in commitment and abs(float(commitment['until_s'])-bound) > 1e-8)):
                return False
        else:
            # Old saved runs predate explicit scope. Their original input's
            # topology is the only authorized frame, never a later row fit.
            direction = np.asarray(topology.get('direction', []), float)
            bound = float(commitment['until_s'])
        sign = commitment['direction']
        return bool(sign in (-1, 1) and direction.shape == (2,) and target.shape == (3,)
            and np.isfinite(np.r_[direction, bound, target]).all()
            and abs(np.linalg.norm(direction)-1.) <= 1e-3
            and sign*(target[:2]@direction-bound) <= .2+1e-8)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def task_specification(mission, spec, home):
    """Which predicates the bound instruction makes mandatory (agreed 2026-10-28).

    The bound command records decide it, never the trajectory: the last accepted
    COVER fixes order and an optional subset of aisles, numbered 1.. from the
    home side exactly as online. The S pattern is required only for S order
    without temporary visits/returns; choosing among aisles only for free order
    over all aisles. Legacy runs without command records keep every predicate.
    """
    commands = mission.get('command_results') if isinstance(mission, dict) else getattr(mission, 'command_results', None)
    n = np.asarray(spec['normal'], float)
    centers = [((c['low_row'][1]+c['high_row'][1])/2) for c in spec['corridors']]
    home_side_first = not centers or np.mean(centers) >= float(np.asarray(home, float)[:2]@n)
    count = len(spec['corridors'])
    indices = [i+1 if home_side_first else count-i for i in range(count)]
    if commands is None:
        return dict(source='legacy_no_command_records', order='S', corridors=None, s_pattern_required=True,
            corridor_choice_required=True, truth_lateral_index=indices, required=lambda index: True)
    accepted = [c for c in commands if c.get('accepted', True)]
    cover = [c for c in accepted if c.get('intent') == 'COVER']
    last = cover[-1] if cover else {}
    order, subset = last.get('order') or 'S', last.get('corridors')
    interrupted = any(c.get('intent') in ('VISIT_CORRIDOR', 'REVISIT_CORRIDOR', 'RETURN_HOME') for c in accepted)
    wanted = None if not subset else set(subset)
    return dict(source='bound_command_records', order=order, corridors=subset, temporary_commands=interrupted,
        s_pattern_required=bool(order == 'S' and not subset and not interrupted),
        corridor_choice_required=bool(order == 'free' and not subset),
        truth_lateral_index=indices, required=lambda index: wanted is None or index in wanted)


def audit_native_decisions(decisions, events=None, policy='qwen', *, required_corridor_count=0,
                           corridor_choice_required=True):
    """Match saved raw model replies, candidate sets and actual goal bindings."""
    if isinstance(decisions, (str, Path)):
        root = Path(decisions)
        records = [dict(input=json.loads(p.read_text()),
            response=json.loads(p.with_name('response.json').read_text()) if p.with_name('response.json').exists() else {})
            for p in sorted((root/'language_decisions').glob('*/input.json'))]
        if events is None:
            path = root/'task_events.jsonl'
            events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        metadata = root/'model_metadata.json'
        model_metadata = json.loads(metadata.read_text()) if metadata.exists() else {}
        config = root/'config.json'
        if config.exists():
            policy = json.loads(config.read_text()).get('coverage', {}).get('policy', policy)
    elif isinstance(decisions, dict):
        records = decisions.get('records', [])
        events = decisions.get('events', events)
        model_metadata = decisions.get('model_metadata', {})
        policy = decisions.get('policy', policy)
    else:
        records, model_metadata = list(decisions), {}
    events = events or []
    # Admission diagnostics preserve an observation even when no inference was
    # requested. A missing response is expected for these inputs.
    observation_only = sum(r.get('input', {}).get('model_requested') is False for r in records)
    records = [r for r in records if r.get('input', {}).get('model_requested') is not False]
    from .coverage_prompt import combined_choice, decision_from_choice
    from .qwen_subgoal import parse_choice, parse_decision
    audited, by_request = [], {}
    for record in records:
        payload, response = record.get('input', {}), record.get('response', {})
        request = payload.get('request_id')
        candidates = payload.get('candidates', [])
        selected, errors = None, []
        decoded = None
        replies = [response]
        try:
            if response.get('response_schema') == 'choice' and payload.get('orders'):
                # Several candidate orders: each raw reply parses to its recorded
                # letter, and the summed letter probabilities give the decision.
                orders, replies = payload['orders'], response.get('orders') or []
                if len(replies) != len(orders):
                    raise ValueError('one reply per candidate order is required')
                for k, (order, reply) in enumerate(zip(orders, replies)):
                    letters = list(order['labels'].values())
                    if len(letters) != len(set(letters)) or set(order['labels']) != set(payload['labels']):
                        raise ValueError('every order must letter the same candidates uniquely')
                    if parse_choice(reply.get('raw')) != reply.get('choice'):
                        errors.append('PARSED_REPLY_DIFFERS_FROM_RAW')
                    if reply.get('request_id') != f'{request}_o{k}':
                        errors.append('REQUEST_ID_MISMATCH')
                pick, _ = combined_choice([(o['labels'], r) for o, r in zip(orders, replies)])
                if pick != (response.get('combined') or {}).get('candidate_id'):
                    errors.append('COMBINED_CHOICE_DIFFERS')
                decoded = decision_from_choice(payload['labels'].get(pick), payload['labels'], candidates)
                if decoded is None:
                    raise ValueError('no offered candidate has letter probability')
            elif response.get('response_schema') == 'choice':
                # point_choice_v1: the raw letter names a candidate through the
                # labels saved with the input; the role fixes the behaviour.
                letters = list((payload.get('labels') or {}).values())
                if len(letters) != len(set(letters)):
                    raise ValueError('candidate letters must be unique')
                choice = parse_choice(response.get('raw'))
                if choice != response.get('choice'):
                    errors.append('PARSED_REPLY_DIFFERS_FROM_RAW')
                decoded = decision_from_choice(choice['choice'], payload['labels'], candidates)
                if decoded is None:
                    raise ValueError('choice is not an offered letter')
            else:
                decoded = parse_decision(response.get('raw'))
            matches = [c for c in candidates if c.get('candidate_id') == decoded['candidate_id']]
            if len(matches) != 1:
                raise ValueError('selected candidate identity must be unique')
            selected = matches[0]
            if response.get('response_schema') != 'choice' and decoded != response.get('decision'):
                errors.append('PARSED_REPLY_DIFFERS_FROM_RAW')
        except (ValueError, TypeError, KeyError, StopIteration):
            errors.append('RAW_REPLY_OR_CANDIDATE_INVALID')
        if response.get('request_id') != request:
            errors.append('REQUEST_ID_MISMATCH')
        if response.get('error'):
            errors.append('MODEL_ERROR')
        model_reply = bool(replies) and all(
            'rule' not in (source := str(reply.get('source', '')).lower()) and 'baseline' not in source
            and (reply.get('input_tokens') or 0) > 0 and (reply.get('output_tokens') or 0) > 0
            and (reply.get('latency_seconds') or 0) > 0 for reply in replies)
        if not model_reply:
            errors.append('NO_MODEL_INFERENCE_EVIDENCE')
        movement = [c for c in candidates if c.get('candidate_id') != 'HOLD' and c.get('role') not in ('HOLD', 'STOP')]
        alternatives = {(c.get('corridor_id'), c.get('direction'), tuple(np.round(c.get('target_world', []), 3))) for c in movement}
        if request in by_request:
            errors.append('DUPLICATE_REQUEST_ID')
        item = dict(request_id=request, model_reply_verified=model_reply, errors=errors, decision=decoded,
            topology=payload.get('state', {}).get('topology', []),
            movement_alternatives=len(alternatives), genuine_multiple_options=len(alternatives) >= 2,
            observed_corridor_options=sorted({c['corridor_id'] for c in movement if c.get('corridor_id')}),
            inspection_corridor_options=sorted({v['corridor_id'] for c in movement
                if (v := _inspection_commitment(c)) is not None}),
            inspection_commitment=_inspection_commitment(selected or {}),
            selected_candidate_id=(selected or {}).get('candidate_id'), selected=selected)
        audited.append(item)
        by_request[request] = item
    bindings, activated, selected_corridors, first_inspections = [], set(), set(), {}
    branch_requests = set()
    for event in events:
        kind = event.get('kind')
        if kind not in ('goal_activated', 'coverage_continuation_activated') or event.get('goal_id') == 'sensor_warmup':
            continue
        request = event.get('source_request_id', event.get('parent_request_id', event.get('request_id')))
        record = by_request.get(request)
        selected = (record or {}).get('selected') or {}
        target = np.asarray(event.get('target_world', []), float)
        candidate_target = np.asarray(selected.get('target_world', []), float)
        valid_target = bool(target.shape == (3,) and candidate_target.shape == (3,)
            and np.isfinite(np.r_[target, candidate_target]).all() and np.linalg.norm(target-candidate_target) <= 1e-5)
        valid_decision = bool(record and event.get('decision') == record['decision'])
        continuation = kind == 'coverage_continuation_activated'
        committed, current = selected.get('commitment') or {}, event.get('commitment') or {}
        role = event.get('role', selected.get('role') if not continuation else None)
        bounded = 'until_s' in committed or 'scope' in committed
        valid_commitment = ('commitment' not in event and not bounded) or _same_commitment(committed, current)
        topology = next((c for c in (record or {}).get('topology', [])
            if c.get('corridor_id') == committed.get('corridor_id')), {})
        if continuation:
            valid_commitment = _same_commitment(committed, current)
            position = np.asarray(event.get('actual_position', []), float)
            direction = np.asarray(topology.get('direction', []), float)
            matches = [p for p in selected.get('route', []) if target.shape == (3,)
                and np.asarray(p.get('target_world', [])).shape == (3,)
                and np.linalg.norm(target-p['target_world']) <= 1e-5]
            outer = committed.get('kind') == 'outer_observation'
            if role is None:
                role = matches[0].get('role') if matches else 'OBSERVE' if outer else (
                    'INSPECT' if committed and event.get('decision', {}).get('behavior') == 'FOLLOW' else None)
            same_intention = bool(committed and valid_commitment and
                (event.get('corridor_id') is None if outer else event.get('corridor_id') == committed.get('corridor_id')))
            forward = bool(same_intention and role == ('OBSERVE' if outer else 'INSPECT')
                and direction.shape == (2,) and position.shape == (3,)
                and target.shape == (3,) and np.isfinite(np.r_[position, direction, target]).all()
                and (target[:2]-position[:2])@direction*committed['direction'] >= -.05)
            if outer and forward:
                normal = np.asarray(topology.get('normal', [-direction[1], direction[0]]), float)
                forward = bool(normal.shape == (2,) and abs(target[:2]@normal-committed['offset_m']) <= .05)
            on_route = any(p.get('corridor_id') == event.get('corridor_id') and p.get('role') == role for p in matches)
            valid_target = on_route or forward
            valid_decision = bool(record and request in activated
                and event.get('parent_candidate_id') == record['selected_candidate_id'])
        else:
            valid_decision &= role == selected.get('role')
        bound_respected = role != 'INSPECT' or _within_inspection_bound(committed, target, topology)
        valid_target &= bound_respected
        verified = bool(record and not record['errors'] and valid_target and valid_decision and valid_commitment)
        if verified and not continuation:
            activated.add(request)
            inspection = record['inspection_commitment']
            if inspection:
                selected_corridors.add(inspection['corridor_id'])
                if len(record['inspection_corridor_options']) >= 2:
                    branch_requests.add(request)
        inspection = (record or {}).get('inspection_commitment') or {}
        inspection_source = bool(verified and inspection and
            inspection.get('corridor_id') == event.get('corridor_id') and
            _same_commitment(committed, current) and current.get('direction') == inspection.get('direction'))
        if role == 'INSPECT':
            cid = event.get('corridor_id')
            first_inspections.setdefault(cid, dict(corridor_id=cid, goal_id=event.get('goal_id'),
                source_request_id=request, verified=inspection_source))
        bindings.append(dict(goal_id=event.get('goal_id'), source_request_id=request,
            verified=verified, continuation=continuation,
            role=role, corridor_id=event.get('corridor_id'), inspection_source_verified=inspection_source,
            commitment_matches_selected_candidate=valid_commitment, inspection_bound_respected=bound_respected,
            target_matches_selected_candidate=valid_target, decision_matches_raw_reply=valid_decision))
    multiple = sum(r['genuine_multiple_options'] for r in audited)
    artifact_passed = bool(policy == 'qwen' and model_metadata.get('base_model') and audited and bindings
        and multiple and all(b['verified'] for b in bindings))
    inspection_sources = bool(first_inspections and all(v['verified'] for v in first_inspections.values()))
    minimum = min(2, required_corridor_count)
    bound_proven = bool(len(selected_corridors) >= minimum and inspection_sources)
    choice_proven = bool(bound_proven and (required_corridor_count <= 1 or branch_requests))
    # Under an S-order specification the constraint layer leaves one useful aisle
    # at each change, so a choice among aisles cannot occur by design; there every
    # first inspection must still be bound to a model decision. Free-order tasks
    # must show the model choosing among at least two inspectable aisles.
    passed = artifact_passed and (not required_corridor_count
        or (choice_proven if corridor_choice_required else bound_proven))
    return dict(schema='native_qwen_decision_provenance_v1', policy=policy, passed=passed,
        artifact_bindings_passed=artifact_passed, required_corridor_count=required_corridor_count,
        corridor_selection_proven=choice_proven, corridor_choice_required=bool(corridor_choice_required),
        corridor_choice_scope=('required: free-order coverage leaves several inspectable aisles'
            if corridor_choice_required else 'reported only: S-order constraints leave one useful aisle per change'),
        model_selected_inspection_corridors=sorted(selected_corridors),
        activated_multi_corridor_requests=sorted(branch_requests),
        all_first_inspections_model_bound=inspection_sources,
        first_inspection_bindings=list(first_inspections.values()),
        model_metadata_present=bool(model_metadata.get('base_model')),
        decisions=len(audited), observation_only_inputs=observation_only, multi_option_decisions=multiple,
        multi_corridor_decisions=sum(len(r['observed_corridor_options']) >= 2 for r in audited),
        activated_goals=len(bindings), bindings=bindings,
        records=[{k:v for k, v in r.items() if k not in ('selected', 'topology')} for r in audited],
        scope='artifact consistency and model provenance; candidate correctness and instruction understanding require paired tests')
