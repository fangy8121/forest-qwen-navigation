"""Task alternatives from observed corridors; no preferred route or truth inputs."""
import numpy as np


def commitment_limit(commitment, corridor):
    """Bound in the current row frame, preserving the selected world plane."""
    scope = commitment.get('scope')
    if scope is None:
        return commitment.get('until_s')
    frame = np.asarray(scope['frame_direction'], float)
    direction, normal = np.asarray(corridor['direction']), np.asarray(corridor['normal'])
    scale = float(direction@frame)
    if scale < .9:
        raise ValueError('commitment frame no longer aligned with observed corridor')
    return (scope['endpoint_s']-corridor['offset_m']*float(normal@frame))/scale


def candidate_intent_metrics(candidate, position, structure, ledger):
    """Describe this option's known commitment, never promise future coverage.

    Intersections use the ledger's persistent projection frame, including after
    a row refit. Transit earns no inspection credit. The caller keeps every
    legal alternative; these measurements do not select or rank a candidate.
    """
    p = np.asarray(position, float)
    route = candidate.get('route') or [candidate]
    points = np.asarray([p]+[point['target_world'] for point in route])
    explicit = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    result = dict(explicit_route_distance_m=round(explicit, 2),
        total_route_distance_m=round(explicit, 2), inspection_range_s_m=None,
        new_coverage_m=None, repeated_coverage_m=None, boundary_observation=None)
    commitment = candidate.get('commitment') or {}
    corridor = next((c for c in structure.get('corridors', [])
        if c['id'] == commitment.get('corridor_id')), None)
    if corridor is None or commitment.get('direction') not in (-1, 1):
        return result
    d, n = np.asarray(corridor['direction']), np.asarray(corridor['normal'])
    sign = commitment['direction']
    try:
        limit = commitment_limit(commitment, corridor)
    except ValueError:
        return result
    end = float(np.clip(limit, *corridor['span']) if limit is not None
                else corridor['span'][1 if sign > 0 else 0])
    last = float(points[-1, :2]@d)
    result['total_route_distance_m'] = round(explicit+max(0., sign*(end-last)), 2)
    if commitment.get('kind') == 'outer_observation':
        side = commitment['side']
        proof = structure.get('outer_sides', {}).get(side, {})
        result['boundary_observation'] = dict(side=side, confirmed=bool(proof.get('confirmed')),
            missing_ray_or_ground_fraction=round(1.-proof.get('observed_fraction', 0.), 3),
            missing_ground_fraction=round(1.-proof.get('terrain_supported_fraction', 0.), 3),
            diverse_views=bool(proof.get('diverse_views')),
            positive_stem_counterexamples=len(proof.get('positive_stem_counterexamples', [])))
        return result
    first = next((i for i, point in enumerate(route) if point['role'] == 'INSPECT'
        and point.get('corridor_id') == corridor['id']), None)
    if first is None:
        return result
    # A direct inspection starts at the measured body; a connection starts at
    # its target aisle's inspection entrance, after all TRANSIT stages.
    start = float((p if first == 0 else points[first+1])[:2]@d)
    required = ledger.required(corridor)
    fixed = np.asarray(ledger.directions[corridor['id']])
    scale, bias = float(d@fixed), float(corridor['offset_m']*(n@fixed))
    if scale < .9:
        return result
    interval = np.array([start, end])*scale+bias
    low, high = sorted(interval)
    low, high = max(low, required[0]), min(high, required[1])
    if sign*(end-start) <= 0:
        high = low

    def overlap(intervals):
        return sum(max(0., min(high, b)-max(low, a)) for a, b in intervals)/scale

    result.update(inspection_range_s_m=np.round(interval, 3).tolist(),
        new_coverage_m=round(overlap(ledger.gaps(corridor)), 2),
        repeated_coverage_m=round(overlap(ledger.covered.get(corridor['id'], [])), 2))
    return result


def distance_before_end(s, end_s, side):
    """Along-aisle distance still to fly to a row end; 0 once past it.

    A change-aisle route first flies 2.2 m past the farther of the two ends,
    so with staggered ends its own headland stage ends more than 8 m beyond
    the current end (run 59fed373: 8.3 m, the route reopened and no change
    was offered again). Past the end the aircraft is in the headland.
    """
    return max(0., (1. if side == 'high' else -1.)*(float(end_s)-float(s)))


def coverage_choices(structure, position, active_corridor, initial_yaw, anchor,
                     probes, *, step=4., observation_yaw=None, bootstrap_budget=32):
    """Enumerate semantic choices. Every route still needs live lower checks.

    ``anchor`` binds XY to an observation-supported height and may reject an
    unsupported location with ValueError. A corridor hypothesis is not a free
    space certificate. Cross-row movement is offered only around confirmed
    ends of both involved corridors.
    """
    p = np.asarray(position, float)
    corridors = sorted(structure.get('corridors', []), key=lambda c:c['offset_m'])
    current = next((c for c in corridors if c['id'] == active_corridor), None)
    unresolved_gaps = structure.get('unresolved_row_gaps', [])
    choices = []
    area = (structure.get('public_work_area') or {}).get('xy_bounds')

    def inside(xy, margin=0.):
        if area is None:
            return True
        xmin, xmax, ymin, ymax = area
        return xmin-margin <= xy[0] <= xmax+margin and ymin-margin <= xy[1] <= ymax+margin

    def add(identifier, route_xy, commitment=None, probe_key=None):
        route = []
        # Outside the declared job nothing is credited or required (run
        # f830a311 spent 250 s flanking the outermost row at y=-0.5).
        if not all(inside(xy, .5) for xy, _, _ in route_xy):
            return False
        try:
            for xy, role, cid in route_xy:
                point = anchor(xy, role, cid)
                if np.linalg.norm(np.asarray(point['target_world'])-p) > .3:
                    route.append(point)
        except ValueError:
            return
        if route:
            value = dict(route[0], candidate_id=identifier, id=identifier,
                label=identifier, kind='observed_coverage_choice', route=route,
                commitment=commitment, probe_key=probe_key)
            choices.append(value)
            return True
        return False

    def local_observations():
        yaw = initial_yaw if observation_yaw is None else observation_yaw
        for index, name in enumerate(('forward', 'forward_left', 'left', 'back_left',
                                      'back', 'back_right', 'right', 'forward_right')):
            angle = yaw+index*np.pi/4
            direction = np.array([np.cos(angle), np.sin(angle)])
            for length in (1.5, .75, .5):
                if add(f'scan_{name}_{int(length*100):03d}',
                        [(p[:2]+length*direction, 'OBSERVE', None)]):
                    choices[-1].update(requires_observed_connection=True,
                        observation_slot='scan_'+name, step_distance=length)

    def inspect(corridor, sign):
        direction, normal = np.asarray(corridor['direction']), np.asarray(corridor['normal'])
        along = float(p[:2]@direction)
        for distance in dict.fromkeys((step, 3., 2., 1.)):
            target = np.clip(along+sign*min(step, distance), *corridor['span'])
            if sign*(target-along) > .35 and add(
                    corridor['id']+('_positive' if sign > 0 else '_negative'),
                    [(direction*target+normal*corridor['offset_m'], 'INSPECT', corridor['id'])],
                    dict(corridor_id=corridor['id'], direction=sign)):
                break

    # A wide pair of observed rows is an observation frontier, not an aisle.
    # Only short fully observed moves are admitted later; no gap is bridged or
    # converted into a traversable corridor by this candidate generator.
    for gap in unresolved_gaps:
        delta = np.asarray(gap['observation_target_xy'])-p[:2]
        distance = np.linalg.norm(delta)
        for length in (1.5, .75, .5):
            if distance > .35 and add(f'{gap["id"]}_observe_{int(length*100):03d}',
                    [(p[:2]+delta*min(1., length/distance), 'OBSERVE', None)],
                    probe_key='frontier:'+gap['id']):
                choices[-1].update(requires_observed_connection=True, frontier_id=gap['id'],
                    observation_slot=gap['id']+'_observe', step_distance=min(length, distance))
    if unresolved_gaps:
        local_observations()

    if not corridors:
        if not unresolved_gaps and probes.get('bootstrap', 0) < bootstrap_budget:
            yaw = initial_yaw if observation_yaw is None else observation_yaw
            for index, name in enumerate(('forward', 'forward_left', 'left', 'back_left',
                                          'back', 'back_right', 'right', 'forward_right')):
                direction = np.array([np.cos(yaw+index*np.pi/4), np.sin(yaw+index*np.pi/4)])
                for length in (1.5, .75, .5):
                    if add('observe_'+name+f'_{int(length*100):03d}',
                            [(p[:2]+length*direction, 'BOOTSTRAP', None)], probe_key='bootstrap'):
                        choices[-1].update(requires_observed_connection=True,
                            observation_slot='bootstrap_'+name, step_distance=length)
        return choices

    # A stale active identity is not used to fabricate a new current corridor.
    if current is None:
        aligned = [c for c in corridors if abs(p[:2]@c['normal']-c['offset_m']) <= .9]
        for c in aligned:
            for sign in (-1, 1):
                inspect(c, sign)
        for c in corridors:
            d, n = np.asarray(c['direction']), np.asarray(c['normal'])
            entry = d*np.clip(p[:2]@d, *c['span'])+n*c['offset_m']
            delta = entry-p[:2]
            distance = np.linalg.norm(delta)
            for length in (1.5, .75, .5):
                if distance > .35 and add(c['id']+f'_entry_observe_{int(length*100):03d}',
                        [(p[:2]+delta*min(1., length/distance), 'OBSERVE', c['id'])]):
                    choices[-1].update(requires_observed_connection=True,
                        observation_slot=c['id']+'_entry_observe', step_distance=min(length, distance))
        # A straight entry toward an outer corridor can cross its bordering row
        # (run b5ce9972: R04_R01 from inside R01_R02); its step then shrinks and
        # disappears. Offer the same corridor around a confirmed end instead:
        # along the current lateral line to the headland, then across outside
        # every row of that corridor. Each anchor keeps the observed-connection
        # check at activation; an unobserved headland reopens the choice.
        for c in corridors:
            d, n = np.asarray(c['direction']), np.asarray(c['normal'])
            lateral = float(p[:2]@n)
            if abs(lateral-c['offset_m']) <= .9:
                continue
            for side, sign in [('low', -1), ('high', 1)]:
                end = c['ends'][side]
                if not end.get('confirmed') or abs(float(p[:2]@d)-end['s']) > 8.:
                    continue
                edge = end['s']+sign*2.2
                if add(c['id']+'_entry_via_'+side,
                        [(d*edge+n*lateral, 'OBSERVE', None), (d*edge+n*c['offset_m'], 'OBSERVE', c['id'])]):
                    choices[-1].update(requires_observed_connection=True, confirmed_ends=[[c['id'], side]],
                        connection_end=side, observation_slot=c['id']+'_entry_via_'+side,
                        connection_distance_m=float(abs(float(p[:2]@d)-end['s'])))
        # Entry is not always a straight reachable connection. Offer short
        # observed moves around occluders; selecting one never authorizes an
        # unknown cross-row connection. Admission keeps the longest supported
        # step in each direction, independently of the task's correct answer.
        if not unresolved_gaps:
            local_observations()
        return choices

    d, n = np.asarray(current['direction']), np.asarray(current['normal'])
    s = float(p[:2]@d)
    for side, sign in [('low', -1), ('high', 1)]:
        end = current['ends'][side]
        inspect(current, sign)
        key = current['id']+':'+side
        if not end.get('confirmed') and probes.get(key, 0) < 4:
            # Observation targets stay near the support frontier; they do not
            # presume that the next row or a traversable connection exists.
            edge = end['s']+sign*(2.+.75*probes.get(key, 0))
            add(current['id']+'_observe_'+side,
                [(d*edge+n*current['offset_m'], 'OBSERVE', current['id'])],
                probe_key=key)

    index = corridors.index(current)
    for neighbor_index in (index-1, index+1):
        if not 0 <= neighbor_index < len(corridors):
            continue
        neighbor = corridors[neighbor_index]
        nd, nn = np.asarray(neighbor['direction']), np.asarray(neighbor['normal'])
        if float(d@nd) < .98:
            continue
        for side, sign in [('low', -1), ('high', 1)]:
            distance_to_end = distance_before_end(s, current['ends'][side]['s'], side)
            if distance_to_end > 8.:
                continue
            if not (current['ends'][side].get('confirmed')
                    and neighbor['ends'][side].get('confirmed')):
                key = neighbor['id']+':connection:'+side
                if current['ends'][side].get('confirmed') and probes.get(key, 0) < 6:
                    edge = sign*max(sign*current['ends'][side]['s'], sign*neighbor['ends'][side]['s'])
                    delta = sign*2.*d if sign*(s-edge) < 1. else (
                        np.sign(neighbor['offset_m']-p[:2]@n)*1.5*n)
                    if add(neighbor['id']+'_connection_observe_'+side,
                            [(p[:2]+delta, 'OBSERVE', current['id'])], probe_key=key):
                        choices[-1].update(requires_observed_connection=True,
                            connection_end=side, connection_distance_m=float(distance_to_end))
                continue
            ends = [current['ends'][side]['s'], neighbor['ends'][side]['s']]
            edge = (min(ends)-2.2) if sign < 0 else (max(ends)+2.2)
            # Always follow the current corridor to its headland before moving
            # laterally. The adjacent corridor is then inspected inward.
            route = [(d*edge+n*current['offset_m'], 'TRANSIT', current['id']),
                     (nd*edge+nn*neighbor['offset_m'], 'TRANSIT', neighbor['id']),
                     (nd*neighbor['span'][0 if sign < 0 else 1]+nn*neighbor['offset_m'],
                      'INSPECT', neighbor['id'])]
            if add(neighbor['id']+'_via_'+side, route,
                    dict(corridor_id=neighbor['id'], direction=-sign)):
                choices[-1]['confirmed_ends'] = [[current['id'], side], [neighbor['id'], side]]
                choices[-1].update(connection_end=side, connection_from_corridor=current['id'],
                    connection_distance_m=float(distance_to_end))

    # Non-adjacent aisles along the headland beyond every row end in between.
    # Adjacent-only changes made a subset or cross-row visit inspect each
    # intermediate aisle in full (run f95290de). Every end on that side must be
    # confirmed, so the headland is bounded by observed row ends; the lower
    # executor still checks each segment.
    for target_index, target in enumerate(corridors):
        if abs(target_index-index) < 2 or float(d@np.asarray(target['direction'])) < .98:
            continue
        between = corridors[min(index, target_index):max(index, target_index)+1]
        td, tn = np.asarray(target['direction']), np.asarray(target['normal'])
        for side, sign in [('low', -1), ('high', 1)]:
            distance_to_end = distance_before_end(s, current['ends'][side]['s'], side)
            if distance_to_end > 8. or not all(c['ends'][side].get('confirmed') for c in between):
                continue
            ends = [c['ends'][side]['s'] for c in between]
            edge = (min(ends)-2.2) if sign < 0 else (max(ends)+2.2)
            route = [(d*edge+n*current['offset_m'], 'TRANSIT', current['id']),
                     (td*edge+tn*target['offset_m'], 'TRANSIT', target['id']),
                     (td*target['span'][0 if sign < 0 else 1]+tn*target['offset_m'], 'INSPECT', target['id'])]
            if add(target['id']+'_via_'+side, route, dict(corridor_id=target['id'], direction=-sign)):
                choices[-1]['confirmed_ends'] = [[c['id'], side] for c in between]
                choices[-1].update(connection_end=side, connection_from_corridor=current['id'],
                    connection_distance_m=float(distance_to_end),
                    skipped_corridors=[c['id'] for c in between[1:-1]])

    # The outer flank is approached around a confirmed headland, never by
    # cutting across a tree row. Both sides remain explicit model alternatives.
    rows = sorted(structure.get('rows', []), key=lambda r:r['offset_m'])
    for side, sign in [('low', -1), ('high', 1)]:
        key = 'outer:'+side
        if (not rows or structure.get('outer_sides', {}).get(side, {}).get('confirmed')
                or probes.get(key, 0) >= 4):
            continue
        row = rows[0 if sign < 0 else -1]
        if row['id'] not in current.get('bordering_rows', []):
            continue
        for end_side, end_sign in [('low', -1), ('high', 1)]:
            end_distance = distance_before_end(s, current['ends'][end_side]['s'], end_side)
            if not current['ends'][end_side].get('confirmed') or end_distance > 8.:
                continue
            edge = current['ends'][end_side]['s']+end_sign*2.2
            offset = row['offset_m']+sign*(3.5+.5*probes.get(key, 0))
            if area is not None:
                # Keep the flank inside the job; a strip too narrow to fly
                # beside the row is observed from the aisle instead.
                low_y, high_y = area[2]+.5, area[3]-.5
                world = d*edge+n*offset
                axis = 1 if abs(n[1]) >= abs(n[0]) else 0
                bound = (low_y, high_y) if axis == 1 else (area[0]+.5, area[1]-.5)
                if not bound[0] <= world[axis] <= bound[1]:
                    target = np.clip(world[axis], *bound)
                    offset += (target-world[axis])/n[axis]
                if sign*(offset-row['offset_m']) < 1.5:
                    continue
            for distance in (4., 3., 2., 1.):
                observe_s = edge-end_sign*distance
                if add('outer_'+side+'_via_'+end_side,
                    [(d*edge+n*current['offset_m'], 'TRANSIT', current['id']),
                     (d*edge+n*offset, 'TRANSIT', None),
                     (d*observe_s+n*offset, 'OBSERVE', None)],
                    dict(kind='outer_observation', side=side, corridor_id=current['id'],
                        direction=-end_sign, offset_m=float(offset)), probe_key=key):
                    choices[-1].update(connection_end=end_side,
                        connection_from_corridor=current['id'],
                        confirmed_ends=[[current['id'], end_side]],
                        connection_distance_m=float(end_distance))
                    break
    return choices


def observed_task_memory(structure, position, yaw, history, candidates):
    """Measured view history and task-neutral geometry features for every option."""
    from .geometry import world_to_local
    p = np.asarray(position, float)
    recent = [{key:row[key] for key in ('candidate_id', 'behavior', 'actual_position',
        'actual_speed_mps', 'completed_at') if key in row} for row in history[-8:]]
    previous = np.asarray([row['actual_position'] for row in recent
        if 'actual_position' in row], float).reshape(-1, 3)
    # The current stopping position adds no revisit information; older nearby
    # observations remain visible so an actual loop gets a small distance.
    previous = previous[np.linalg.norm(previous-p, axis=1) > .25]
    xy = np.asarray([stem['xy'] for stem in structure.get('stem_candidates', [])], float).reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    stem_count = len(xy)
    source = 'detected_stem_candidates'
    if not len(xy):
        xy = np.asarray(structure.get('structure_points_xy', []), float).reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(axis=1)]
        source = 'observed_height_band_returns_not_confirmed_trees'
    summary = dict(count=len(xy), kind=source,
        source='online point measurements only; no true tree positions')
    features = {}
    if len(xy):
        world = np.c_[xy, np.full(len(xy), p[2])]
        body = world_to_local(world, p, yaw)[:, :2]
        centroid, median = np.mean(xy, axis=0), np.median(xy, axis=0)
        center_body = np.mean(body, axis=0)
        ranges = np.linalg.norm(xy-p[:2], axis=1)
        summary.update(centroid_world_xy_m=np.round(centroid, 2).tolist(),
            median_world_xy_m=np.round(median, 2).tolist(),
            centroid_body_xy_m=np.round(center_body, 2).tolist(),
            median_body_xy_m=np.round(np.median(body, axis=0), 2).tolist(),
            centroid_bearing_body_deg=round(float(np.rad2deg(np.arctan2(center_body[1], center_body[0]))), 1),
            median_range_m=round(float(np.median(ranges)), 2),
            nearest_world_xy_m=np.round(xy[np.argsort(ranges)[:16]], 2).tolist(),
            nearest_body_xy_m=np.round(body[np.argsort(ranges)[:16]], 2).tolist())
    for c in candidates:
        target = np.asarray(c['target_world'], float)
        value = dict(nearest_previous_observation_m=(round(float(np.linalg.norm(
            previous-target, axis=1).min()), 2) if len(previous) else None))
        if len(xy):
            distances = np.linalg.norm(xy-target[:2], axis=1)
            value.update(median_structure_range_m=round(float(np.median(distances)), 2),
                median_structure_range_change_m=round(float(np.median(distances)-np.median(ranges)), 2),
                nearest_structure_range_m=round(float(distances.min()), 2))
        features[c['candidate_id']] = value
    return dict(recent_completed_actual_goals=recent, observed_structure=summary,
        detected_stem_count=stem_count,
        history_scope='last8 actual goal endpoints, position distance only'), features


def lateral_reference(corridors, origin=None):
    """Shared lateral axis; with ``origin`` (home XY) it points away from home."""
    corridors = [c for c in corridors if c.get('normal') is not None and c.get('offset_m') is not None]
    if not corridors:
        return None
    reference = np.asarray(corridors[0]['normal'], dtype=float)[:2]
    if origin is not None:
        offsets = [float(c['offset_m'])*(1. if np.asarray(c['normal'], float)[:2]@reference >= 0 else -1.)
                   for c in corridors]
        if np.mean(offsets) < float(np.asarray(origin, float)[:2]@reference):
            reference = -reference
    return reference


def corridor_lateral_order(corridors, origin=None):
    """Lateral order of observed corridors, independent of their online IDs.

    Online row IDs follow discovery order (an outermost row found late may be
    ``R04`` next to ``R01``), so IDs cannot express adjacency. All corridors are
    projected on one shared normal; ``lateral_index`` is 1-based and
    ``observed_edge`` marks the outermost *observed* corridor on each side.
    Undiscovered corridors may still lie further out. With ``origin`` (home XY),
    index 1 is the home side, so "aisle 1" in an instruction is the aisle nearest
    take-off; the independent evaluation numbers true aisles the same way.
    """
    corridors = [c for c in corridors if c.get('normal') is not None and c.get('offset_m') is not None]
    if not corridors:
        return {}
    reference = lateral_reference(corridors, origin)
    def lateral(c):
        normal = np.asarray(c['normal'], dtype=float)[:2]
        return float(c['offset_m'])*(normal@reference)
    ordered = sorted(corridors, key=lateral)
    result = {}
    for index, c in enumerate(ordered, start=1):
        edge = ('low' if index == 1 else None) if len(ordered) > 1 else 'only'
        if index == len(ordered) and len(ordered) > 1:
            edge = 'high'
        result[c['id']] = dict(lateral_index=index, observed_edge=edge)
    return result


def compact_prompt_state(state):
    """Preserve every task object and interval, compress repeated candidate keys.

    The canonical audit state stays unchanged. Numeric display is rounded to
    three decimals; exact bindings, ledger and geometry remain outside Qwen.
    """
    def rounded(value):
        if isinstance(value, dict):
            return {key:rounded(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [rounded(item) for item in value]
        return round(value, 3) if isinstance(value, float) else value

    projected = rounded(state)
    coverage = projected.get('coverage', {})
    for corridor in coverage.get('corridors', []):
        for key in ('required', 'covered', 'uncovered'):
            if key in corridor:
                corridor[key+'_mm'] = [[int(round(endpoint*1000)) for endpoint in interval]
                    for interval in corridor.pop(key)]
    corridors = coverage.pop('corridors', [])
    if corridors:
        columns = list(dict.fromkeys(key for corridor in corridors for key in corridor))
        coverage.update(corridor_columns=columns,
            corridor_rows=[[corridor.get(key) for key in columns] for corridor in corridors])
    candidates = projected.pop('candidates', [])
    for candidate in candidates:
        scope = candidate.get('route_scope', [])
        if len(scope) > 4:
            runs = []
            for role, corridor in scope:
                if runs and runs[-1][:2] == [role, corridor]:
                    runs[-1][2] += 1
                else:
                    runs.append([role, corridor, 1])
            if len(runs)*3 < len(scope)*2:
                candidate['route_scope_runs'] = runs
                del candidate['route_scope']
        for key, replacement in (('target_world', 'target_world_mm'),
                ('relative_body_xyz_m', 'relative_body_xyz_mm'),
                ('inspection_range_s_m', 'inspection_range_s_mm')):
            if key in candidate:
                value = candidate.pop(key)
                candidate[replacement] = ([int(round(v*1000)) for v in value] if value is not None else None)
        for key in ('explicit_route_distance_m', 'total_route_distance_m',
                    'new_coverage_m', 'repeated_coverage_m'):
            if key in candidate:
                value = candidate.pop(key)
                candidate[key[:-2]+'_mm'] = int(round(value*1000)) if value is not None else None
    def flatten(record, prefix=''):
        values = {}
        for key, value in record.items():
            name = prefix+key
            if isinstance(value, dict):
                values.update(flatten(value, name+'.'))
            else:
                values[name] = value
        return values

    scopes, flat = [], []
    for candidate in candidates:
        scope = candidate.pop('commitment_scope', None)
        reference = None
        if scope is not None:
            if scope not in scopes:
                scopes.append(scope)
            reference = 'S'+str(scopes.index(scope))
        flat.append(flatten(dict(candidate, scope_ref=reference)))
    if scopes:
        scope_records = [flatten(dict(scope_ref='S'+str(index), **scope))
            for index, scope in enumerate(scopes)]
        scope_columns = sorted({key for row in scope_records for key in row})
        projected['scope_columns'] = scope_columns
        projected['scope_rows'] = [[row.get(key) for key in scope_columns] for row in scope_records]
    # Shared fields remain explicit once, without repeating them in every row.
    defaults = {key:flat[0].get(key) for key in {key for c in flat for key in c}
        if all(c.get(key) == flat[0].get(key) for c in flat)}
    if defaults:
        projected['candidate_defaults'] = defaults
        flat = [{key:value for key, value in c.items() if key not in defaults} for c in flat]
    principal = ['candidate_id', 'role', 'corridor_id', 'inspection_direction',
        'target_world_mm', 'relative_body_xyz_mm', 'observation_slot', 'probe', 'frontier_id']
    all_keys = {key for c in flat for key in c}
    columns = [key for key in principal if key in all_keys]+sorted(all_keys-set(principal))
    projected['candidate_columns'] = columns
    projected['candidate_rows'] = [[c.get(key) for key in columns] for c in flat]
    return projected


def discovery_phase_context(state):
    """Explain the current phase and every option; never rank or select an ID."""
    spec = state.get('task_spec') or {}
    if state['intent'] == 'VISIT_CORRIDOR':
        target = spec.get('target_corridor')
        index = spec.get('target_lateral_index') or (spec.get('target') or {}).get('lateral_index')
        lines = [f'当前阶段VISIT：按指令前往横向序号{index}的通道{target or "（尚未观测到）"}并巡检其未巡部分；'
                 '允许离开未巡完的当前通道，换行只保留朝目标方向的选项。'
                 +('完成后回到原任务断点继续。' if state.get('resume_after') else '完成后悬停等待指令。')]
        if state.get('current_corridor_lateral_index') is not None:
            lines.append(f'当前通道{state.get("current_corridor")}横向序号'
                         f'{state["current_corridor_lateral_index"]}/{state.get("observed_corridor_count")}。')
        for c in state['candidates']:
            new = c.get('new_coverage_m')
            amount = '巡检收益不适用' if new is None else f'可补巡{new:g}米'
            lateral = (f'，目标通道横向序号{c["corridor_lateral_index"]}'
                       if c.get('corridor_lateral_index') is not None else '')
            lines.append(f'{c["candidate_id"]}：behavior={c.get("allowed_behavior")}，{amount}{lateral}。')
        lines.append('严格输出卡片所列candidate_id和behavior。')
        return 'VISIT', '\n'.join(lines)
    if state['intent'] != 'COVER':
        return state['intent'], ''
    if state.get('coverage_result') == 'COMPLETE_UNDER_ASSUMPTIONS':
        lines = ['当前阶段RETURN：作业区内全部通道已巡完，行末和外侧均已确认，按指令返回起点并停稳。'
                 'RETURN_HOME沿已飞路径返回；RETURN_HOME_HEADLAND经全部行末外侧空地返回，路程通常更短。'
                 '两者都由下层实时检查，受阻可HOLD观察。']
        for c in state['candidates']:
            route = c.get('explicit_route_distance_m')
            lines.append(f'{c["candidate_id"]}：behavior={c.get("allowed_behavior")}'
                         +(f'，路线{route:g}米' if route is not None else '')+'。')
        lines.append('严格输出卡片所列candidate_id和behavior。')
        return 'RETURN', '\n'.join(lines)
    remaining = state.get('current_corridor_uncovered_m')
    recover = state.get('recovery_pending', False)
    completed_end = state.get('at_confirmed_end') and remaining is not None and remaining <= .05
    # Covered but the near end is not yet confirmed: no cross-row option exists
    # until it is (run f830a311 flew the finished aisle back instead).
    confirm_end = (not completed_end and remaining is not None and remaining <= .05
                   and state.get('unconfirmed_near_end') is not None)
    if recover or completed_end or state.get('current_corridor') is not None:
        phase = ('RECOVER' if recover else 'CHANGE_OR_REPAIR' if completed_end else
                 'CONFIRM_END' if confirm_end else 'INSPECT_CORRIDOR')
        description = ('blocked_goal执行受阻，按新观测选择恢复。' if recover else
            '当前通道已巡完，应换行、补巡或补查边界。' if completed_end else
            f'当前通道已巡完，需确认行末{state["unconfirmed_near_end"]}：优先选择observe_'
            f'{state["unconfirmed_near_end"]}或connection_observe观测行末，确认后才会出现换行选项；'
            '反向重飞已巡通道没有新覆盖。' if confirm_end else
            '当前通道尚未巡完；比较各方向实际可补巡区间，保持S形任务连续性。')
        lines = [f'当前阶段{phase}：{description}'
            '整廊未巡量不能区分方向；卡片补巡/重复只指该动作的已知承诺，不是全图收益。']
        if spec.get('corridors'):
            lines.append(f'任务规格：只巡横向序号{spec["corridors"]}的通道，其余通道只作转场。')
        if spec.get('order') == 'free':
            lines.append('任务规格：不要求S形顺序，可在未巡完时换行，但不要重复已巡完的通道。')
        if state.get('current_corridor_lateral_index') is not None:
            lines.append(f'当前通道{state.get("current_corridor")}横向序号'
                f'{state["current_corridor_lateral_index"]}/{state.get("observed_corridor_count")}。')
        previous = state.get('previous_intention')
        if previous:
            lines.append(f'上一实际意图：通道{previous.get("corridor_id")}，方向{previous.get("direction")}，'
                f'结束端{previous.get("actual_end")}；当前端{state.get("current_end")}。')
        for c in state['candidates']:
            new = c.get('new_coverage_m')
            amount = '巡检收益不适用' if new is None else f'可补巡{new:g}米'
            proof = c.get('boundary_observation')
            if proof:
                amount += f'，补查外侧{proof["side"]}，尚缺观测{proof["missing_ray_or_ground_fraction"]:.1%}'
            behavior = c.get('allowed_behavior') or {'TRANSIT':'CHANGE_AISLE', 'INSPECT':'FOLLOW', 'OBSERVE':'FOLLOW',
                'BOOTSTRAP':'FOLLOW', 'HOLD':'OBSERVE', 'RETURN':'RETURN', 'REJOIN':'REJOIN'}[c['role']]
            connection = (f'，端{c["connection_end"]}距{c["connection_distance_m"]:g}米'
                if c.get('connection_end') is not None and c.get('connection_distance_m') is not None else '')
            lateral = ''
            if c.get('corridor_lateral_index') is not None:
                lateral = f'，目标通道横向序号{c["corridor_lateral_index"]}/{state.get("observed_corridor_count")}'
                if c.get('corridor_observed_edge') in ('low', 'high', 'only'):
                    lateral += '（已观测最外侧）'
            if c.get('skipped_lateral_indices'):
                lateral += (f'，经行末外侧跨过横向序号{c["skipped_lateral_indices"]}'
                            +(f'（其中未巡{c["skipped_uncovered_m"]:g}米，之后需返回补巡）'
                              if c.get('skipped_uncovered_m') else '（均已巡完）'))
            lines.append(f'{c["candidate_id"]}：behavior={behavior}，{amount}{connection}{lateral}。')
        lines.append('TRANSIT经确认行末连接目标通道后巡检；重定位、绕障和补证仍可必要，首段短不代表全程短。'
            '严格输出卡片所列candidate_id和behavior。')
        return phase, '\n'.join(lines)
    if state.get('topology'):
        return 'ENTER_CORRIDOR', '当前阶段ENTER_CORRIDOR：先通过有观测依据的入口进入通道；尚未绑定当前通道。'
    lines = ['当前阶段FIND_CORRIDOR：尚未识别出可巡通道。当前任务是接近观测结构、获得树行证据。'
        'height_band_returns是较弱的点云依据，不等于已确认树木；没有结构观测时可用公开初始朝向作为观察参考。']
    observed = state.get('observation_memory', {}).get('observed_structure', {})
    if observed.get('count'):
        lines.append(f'观测结构在机体坐标{observed.get("centroid_body_xy_m")}米，'
            f'相对机头方位{observed.get("centroid_bearing_body_deg")}度；x正向前、y正向左。')
    for c in state['candidates']:
        geometry = c.get('geometry', {})
        delta = geometry.get('median_structure_range_change_m')
        relation = ('观测结构距离未知' if delta is None else
            ('更接近' if delta < -.02 else '更远离' if delta > .02 else '基本不变')
            +f'观测结构，距离变化{delta:+.2f}米')
        previous = geometry.get('nearest_previous_observation_m')
        history = '旧视点距离无记录' if previous is None else f'距旧视点最近{previous:.2f}米'
        authority = '，当前任务未授权返航' if c.get('authorized_now') is False else ''
        lines.append(f'{c["candidate_id"]}：机体移动{c["relative_body_xyz_m"]}米；{relation}；{history}{authority}。')
    lines.append('比较全部候选的观测收益。通常应接近结构并增加观察基线，避免在远离结构的空地循环；'
        '必要的绕障或恢复例外由当前图像和几何信息判断。输出所选真实candidate_id和behavior。')
    return 'FIND_CORRIDOR', '\n'.join(lines)
