"""Offline decision test set for point_choice_v1.

An item is one saved decision (real run or synthetic scene) with the set of
acceptable candidates from an explicit task teacher. Images are redrawn for
several candidate orders so that letter-position bias shows up as errors.
"""
from copy import deepcopy
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np

from .coverage_prompt import ROUTE_IDS, assign_labels, candidate_orders

MOVE_ROLES = ('INSPECT', 'TRANSIT', 'OBSERVE', 'BOOTSTRAP', 'RETURN', 'REJOIN')
MIN_GAIN = .5


def target_aisle(candidate, topology):
    """Lateral index of the aisle a candidate leads to, or None."""
    commitment = candidate.get('commitment') or {}
    if commitment.get('kind') == 'outer_observation':
        return None
    cid = commitment.get('corridor_id') or candidate.get('corridor_id')
    if cid is None:
        match = ROUTE_IDS.match(candidate['candidate_id'])
        name = candidate['candidate_id']
        cid = match.group('cid') if match else (name.split('_entry_observe_')[0] if '_entry_observe_' in name else None)
    return (topology.get(cid) or {}).get('lateral_index')


def teacher(state, candidates, public):
    """Acceptable candidate ids and a category, or (None, reason) if undefined.

    S starts at the home-side outermost open aisle, finishes the current aisle,
    then changes to an adjacent open aisle (else the nearest). A subset keeps
    to its aisles; free order accepts any aisle that adds coverage. A visit
    goes to its target aisle. After coverage, any return is accepted.
    """
    topology = {t['corridor_id']: t for t in state.get('topology', [])}
    by_index = {t['lateral_index']: t for t in topology.values()}
    spec = state.get('task_spec') or {}
    uncovered = {row['corridor_id']: row['uncovered_length_m'] for row in state['coverage']['corridors']}
    required = set(spec.get('corridors') or by_index)
    open_aisles = {i for i, t in by_index.items() if i in required and (uncovered.get(t['corridor_id']) or 0.) > MIN_GAIN}
    moves = [c for c in candidates if c['role'] in MOVE_ROLES]
    gain = {c['candidate_id']: (public.get(c['candidate_id']) or {}).get('new_coverage_m') for c in moves}

    def ids(select):
        return [c['candidate_id'] for c in moves if select(c)]

    if state.get('intent') == 'COVER' and (state.get('completion') or {}).get('eligible'):
        return ids(lambda c: c['role'] == 'RETURN'), 'return'
    if state.get('intent') == 'VISIT_CORRIDOR':
        k = spec.get('target_lateral_index')
        if k is None:
            return None, 'visit target unresolved'
        return ids(lambda c: target_aisle(c, topology) == k), 'visit'
    if state.get('intent') != 'COVER' or not open_aisles:
        return None, 'no open aisle'
    free = spec.get('order') == 'free'
    current = (topology.get(state.get('current_corridor')) or {}).get('lateral_index')
    if current is None:
        goal = open_aisles if free else {min(open_aisles)}
        found = ids(lambda c: target_aisle(c, topology) in goal and c['role'] != 'RETURN'
                    and (c['role'] not in ('INSPECT', 'TRANSIT') or (gain[c['candidate_id']] or 0.) > MIN_GAIN))
        return found, 'start'
    if current in open_aisles:
        here = ids(lambda c: c['role'] == 'INSPECT' and target_aisle(c, topology) == current
                   and (gain[c['candidate_id']] or 0.) > MIN_GAIN)
        if here:
            return here, 'inspect'
    if free:
        goal = open_aisles
    else:
        nearest = min(abs(i-current) for i in open_aisles)
        goal = {i for i in open_aisles if abs(i-current) == nearest}
    changes = ids(lambda c: c['role'] == 'TRANSIT' and target_aisle(c, topology) in goal)
    category = 'subset' if spec.get('corridors') else 'free' if free else 'change'
    if changes:
        return changes, category
    # No change offered yet: the connection or row end must be observed first.
    observe = ids(lambda c: c['role'] == 'OBSERVE' and re.search(r'_(connection_observe|observe)_(low|high)$',
                                                                  c['candidate_id']))
    return (observe, category+'_observe') if observe else (None, 'no acceptable move offered')


def permutations(candidates, count):
    """Candidate orders: rotations of the movement options, HOLD/STOP last (as online)."""
    return candidate_orders(candidates, count)


def render_item(source, labels, output, context, global_view=None):
    """Redraw both images of a saved decision for one labelling."""
    from .coverage_images import render_bev, render_global
    from .subgoal_inputs import _safety
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads((source/'input.json').read_text())
    structure = json.loads((source/'structure.json').read_text())
    replay = np.load(source/'replay.npz')
    if structure.get('overview') is not None:
        structure['overview'] = dict(structure['overview'], flags=replay['overview'])
    state = payload['state']
    position, yaw = np.asarray(state['position_world'], float), float(state['yaw_world_rad'])
    order = {t['corridor_id']: dict(lateral_index=t.get('lateral_index'), observed_edge=t.get('observed_edge'))
             for t in state.get('topology', [])}
    trail = replay['trail']
    render_global(structure, state['coverage'], order, position, yaw, trail[0], trail, payload['candidates'],
                  labels, output/'global.png', **(global_view or {}))
    flight = SimpleNamespace(**context['flight'])
    safety = _safety(dict(points=replay['points'], evidence=replay['evidence']), position, yaw, flight,
                     context['operating_bounds'], context['height_bounds']
                     ).with_clearance_reserve(context.get('planning_clearance_reserve', 0.))
    ground = None
    if context.get('terrain'):
        from forest_generator.terrain import grid_height
        t = {k: np.asarray(v, float) for k, v in context['terrain'].items()}

        def ground(xy):
            xy = np.asarray(xy, float).reshape(-1, 2)
            return grid_height(xy, t['xs'], t['ys'], t['heights']), np.ones(len(xy), bool)
    render_bev(safety, position, yaw, payload['candidates'], labels, output/'bev.png', ground=ground)
    return [str(output/'global.png'), str(output/'bev.png')]


def build_item(source, item_id, output, context, count=3, category_hint=None, global_view=None):
    """One test item with ``count`` labelled image pairs, or None with a reason."""
    payload = json.loads((Path(source)/'input.json').read_text())
    if payload.get('protocol') != 'point_choice_v1':
        return None, 'not point_choice_v1'
    state, candidates = payload['state'], payload['candidates']
    public = {p['candidate_id']: p for p in state.get('candidates', [])}
    acceptable, category = teacher(state, candidates, public)
    moves = [c['candidate_id'] for c in candidates if c['role'] not in ('HOLD', 'STOP')]
    if not acceptable:
        return None, category
    if len(moves) < 2:
        return None, 'single move'
    output = Path(output)
    variants = []
    for k, ordered in enumerate(permutations(candidates, count)):
        labels = assign_labels(ordered)
        images = render_item(source, labels, output/f'p{k}', context, global_view)
        variants.append(dict(labels=labels, images=images, order=[c['candidate_id'] for c in ordered]))
    item = dict(id=item_id, source=str(source), category=category_hint or category, teacher_category=category,
                command=state['command'], acceptable=acceptable, moves=moves,
                trivial=set(moves) <= set(acceptable), rule_reference=payload.get('rule_reference'),
                variants=variants)
    (output/'item.json').write_text(json.dumps(item, ensure_ascii=False, indent=1))
    return item, category


# --- synthetic scenes ----------------------------------------------------------------

class _Structure:
    last_error = None

    def __init__(self, data):
        self.data = data

    def snapshot(self):
        return deepcopy(self.data)

    def close(self):
        pass


def synthetic_structure(aisles=6, spacing=5., x0=24., length=66., y0=3., seed=0, unconfirmed=()):
    """Parallel plantation with ``aisles`` aisles; rows every ``spacing`` m from ``y0``."""
    from .online_structure import OVERVIEW_BAND, OVERVIEW_FREE, OVERVIEW_RESOLUTION
    rng = np.random.default_rng(seed)
    ys = y0+spacing*np.arange(aisles+1)
    rows, stems = [], []
    for i, y in enumerate(ys):
        xs = np.arange(x0, x0+length+1e-6, 2.5)+rng.normal(0., .25, len(np.arange(x0, x0+length+1e-6, 2.5)))
        support = np.c_[xs, y+rng.normal(0., .15, len(xs))]
        rows.append(dict(id=f'R{i+1:02d}', direction=[1., 0.], normal=[0., 1.], offset_m=float(y),
                         spans=[[[float(xs[0]), float(y)], [float(xs[-1]), float(y)]]], support=len(xs),
                         support_xy=support.tolist()))
        stems += [dict(xy=p.tolist(), radius_m=float(rng.uniform(.15, .3))) for p in support]
    corridors = []
    for i, (a, b) in enumerate(zip(rows, rows[1:])):
        cid = a['id']+'_'+b['id']
        corridors.append(dict(id=cid, direction=[1., 0.], normal=[0., 1.], offset_m=float((ys[i]+ys[i+1])/2),
            span=[float(x0), float(x0+length)], bordering_rows=[a['id'], b['id']],
            ends={side: dict(s=float(s), confirmed=(i+1, side) not in unconfirmed)
                  for side, s in (('low', x0), ('high', x0+length))}))
    bounds = [0., x0+length+25., 0., float(ys[-1]+3.)]
    origin = np.array([bounds[0], bounds[2]])-4.-OVERVIEW_RESOLUTION/2
    shape = np.ceil((np.array([bounds[1]-bounds[0], bounds[3]-bounds[2]])+8.)/OVERVIEW_RESOLUTION).astype(int)+1
    flags = np.zeros(shape, np.uint8)
    flags[:] = OVERVIEW_FREE
    cells = np.floor((np.asarray([s['xy'] for s in stems])-origin)/OVERVIEW_RESOLUTION).astype(int)
    for du in (0, 1):
        for dv in (0, 1):
            flags[np.clip(cells[:, 0]+du, 0, shape[0]-1), np.clip(cells[:, 1]+dv, 0, shape[1]-1)] |= OVERVIEW_BAND
    return dict(rows=rows, corridors=corridors, stem_candidates=stems,
                outer_sides={side: dict(confirmed=True) for side in ('low', 'high')},
                public_work_area=dict(xy_bounds=bounds), unresolved_row_gaps=[], identity_ambiguous=False,
                overview=dict(origin=origin.tolist(), resolution=OVERVIEW_RESOLUTION, flags=flags))


def local_snapshot(structure, position, ground=0.):
    """Observed hits (stem columns) and free evidence around the vehicle."""
    position = np.asarray(position, float)
    stems = np.asarray([s['xy'] for s in structure['stem_candidates']])
    near = stems[(np.abs(stems-position[:2]) <= [6., 5.]).all(axis=1)]
    angles = np.linspace(0., 2*np.pi, 8, endpoint=False)
    heights = np.arange(ground+.2, ground+6., .2)
    points = np.array([[x+.15*np.cos(a), y+.15*np.sin(a), z] for x, y in near for a in angles for z in heights])
    grid = np.stack(np.meshgrid(np.arange(-5., 5.01, .2), np.arange(-4., 4.01, .2), np.arange(-1., 1.01, .2),
                                indexing='ij'), -1).reshape(-1, 3)+position
    if len(near):
        from scipy.spatial import cKDTree
        grid = grid[cKDTree(near).query(grid[:, :2])[0] > .35]
    return dict(timestamp=1., captured=time.perf_counter(),
                points=points.reshape(-1, 3) if len(points) else np.array([[1e3, 1e3, 0.]]), evidence=grid)


def synthetic_decision(folder, *, aisles=6, covered=(), partial=None, current=None, end=None, yaw=0.,
                       position=None, order='S', subset=None, intent='COVER', visit=None, seed=0,
                       unconfirmed=(), command=None, flight=None, previous=None):
    """Run the real mission planner on a synthetic plantation; write one decision.

    ``covered``: fully inspected lateral indices; ``partial``: (index, s_from, s_to)
    covered part of one aisle; ``current``/``end``: the vehicle's aisle and end
    ('low'/'high'/None = mid-aisle at ``position``); ``previous``: the inspection
    direction (+1/-1) just finished in ``current``.
    """
    from .coverage_mission import CoverageMission
    from .coverage_flight import RuleWorker
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = synthetic_structure(aisles, seed=seed, unconfirmed=unconfirmed)
    corridors = {i+1: c for i, c in enumerate(data['corridors'])}
    home = np.array([corridors[1]['span'][0]-4.3, corridors[1]['offset_m'], 2.5])
    if position is None:
        if current is None:
            position = home.copy()
        else:
            c = corridors[current]
            s = c['span'][0] if end == 'low' else c['span'][1]
            position = np.array([s, c['offset_m'], 2.5])
    position = np.asarray(position, float)
    terrain = dict(xs=np.arange(-10., 140., 5.), ys=np.arange(-10., 60., 5.), heights=np.zeros((30, 14)))
    flight = flight or dict(height=2.0, max_speed=1.8, max_acceleration=1.0, max_jerk=4.0, max_yaw_rate=1.0,
        max_yaw_acceleration=2.0, corner_distance=1.5, max_path_deviation=.3, tracking_limit=1.0,
        settle_seconds=3.0, settle_speed=.15, settle_error=.15)
    text = command or {'S': '沿S形逐条巡检整个树林，在已确认的行末进入相邻未巡通道反向巡检，补齐遗漏，完成后返回起点并停稳。',
                       'free': '按任意顺序巡检全部通道，完成后返回起点并停稳。'}[order]
    m = CoverageMission(RuleWorker(), text, home.tolist(), 0., SimpleNamespace(**flight), folder,
        [0., 125., -15., 55.], [-5., 20.], _Structure(data), decision_policy='qwen', desired_agl=2.5,
        max_decisions=100, decision_input='point_choice', terrain_truth=terrain, planning_clearance_reserve=.15)
    try:
        m.status, m.command_ready = 'WAITING_NEXT', True
        m.intent = intent
        m.task_spec = dict(order=order if intent == 'COVER' else None, corridors=subset,
                           target=dict(lateral_index=visit) if visit else None,
                           target_corridor=corridors[visit]['id'] if visit else None)
        for i in covered:
            m.ledger.covered[corridors[i]['id']] = [list(corridors[i]['span'])]
            m.ledger.directions[corridors[i]['id']] = [1., 0.]
        if partial is not None:
            i, a, b = partial
            m.ledger.covered[corridors[i]['id']] = [[a, b]]
            m.ledger.directions[corridors[i]['id']] = [1., 0.]
        if current is not None:
            m.active_corridor = corridors[current]['id']
            if previous is not None:
                m.last_completed_intention = dict(corridor_id=corridors[current]['id'], direction=previous,
                    actual_end=end, end_confirmed=True, actual_position=position.tolist(), time=1.,
                    task_revision=1)
        # Flown path: the covered aisles in S order from the home side.
        trail, side = [home.tolist()], 0
        for i in sorted(covered):
            c = corridors[i]
            for s_end in (c['span'][side], c['span'][1-side]):
                trail.append([s_end, c['offset_m'], 2.5])
            side = 1-side
        trail.append(position.tolist())
        m.breadcrumbs = trail
        m.latest_position = position
        snapshot = local_snapshot(data, position)
        m.latest, m.latest_yaw = snapshot, yaw
        m._plan(position)
        request = folder/'decision'
        request.mkdir(exist_ok=True)
        m._prepare_native(snapshot, position, yaw, 'decision_000', request, None)
        return request, dict(flight=flight, operating_bounds=[0., 125., -15., 55.], height_bounds=[-5., 20.],
                             planning_clearance_reserve=.15,
                             terrain={k: np.asarray(v).tolist() for k, v in terrain.items()})
    finally:
        m.close()
