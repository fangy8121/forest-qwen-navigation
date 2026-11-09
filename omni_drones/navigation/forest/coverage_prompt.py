"""Label-only coverage decisions: candidate selection, letters and short prompt.

Qwen sees two images and a short text and answers one letter; coordinates,
behaviour and completion come from the chosen candidate. Directions in text
use the world-aligned image frame (right = +x, up = +y).
"""
import re
import string

import numpy as np

PROTOCOL = 'point_choice_v1'
BEHAVIOR = {'INSPECT': 'FOLLOW', 'BOOTSTRAP': 'FOLLOW', 'OBSERVE': 'FOLLOW', 'TRANSIT': 'CHANGE_AISLE',
            'RETURN': 'RETURN', 'REJOIN': 'REJOIN', 'HOLD': 'OBSERVE', 'STOP': 'REQUEST_STOP'}
PROGRESS = ('INSPECT', 'TRANSIT', 'REJOIN', 'RETURN')
MAX_MOVES = 8
# Two observation moves ending this close (XY) are one point on the images.
SAME_PLACE = .6
# Aisle changes/inspections adding no more than this are dropped while another
# option adds coverage (run 79f68b73: a change to a finished aisle cost 140 s).
MIN_GAIN = .5
LEGEND = ('图1全局图（向右为x正向，向上为y正向）：灰=未观测，浅色=已观测，深绿=树干和枝叶，蓝线=已巡，橙线=未巡，'
          '实心/空心圆=行末已确认/未确认，方框数字=通道序号（从起点一侧数），黑方块=起点，红三角=无人机及朝向，'
          '黑框=图2范围，紫线和紫圈字母=候选路线和终点。'
          '图2局部图（与图1同向，每格1米）：橙=障碍，灰=未知，浅色=已观测可飞，紫圈字母=候选目标，白底字母=图外目标的方向。')
# Repeated next to the answer format for a whole-forest S task: in the task line
# alone it fixed 1 of 5 live misses, here 3 (runs 853c339a, b1accde0).
S_RULE = 'S形：所在通道没巡完时先沿它巡完，再经行末换到相邻通道；从起点一侧的第1条开始。'
RULES = ('规则：按指令自主决定通道顺序、方向、补巡和返航；换行只经已确认的行末；任务未完成时不提供返航；'
         '信息不足时可选观察或悬停。先按指令判断下一步应该巡检第几条通道（不巡检时填null），'
         '再选能做到这件事的候选。只输出JSON：{"next_aisle":序号,"choice":"候选字母"}，不要解释。')
INTENTS = dict(COVER='巡检', VISIT_CORRIDOR='去指定通道巡检', REVISIT_CORRIDOR='回到刚才的通道巡检',
               RETURN_HOME='返回起点', PAUSE='暂停', RESUME='继续', FINISH='任务完成，停稳结束')
ROUTE_IDS = re.compile(r'^(?P<cid>.+?)_(?P<kind>via|entry_via|connection_observe|observe)_(?P<side>low|high)$')


def _returning(c):
    return c.get('requested_intent') == 'RETURN_HOME'


def _end(c):
    return np.asarray((c.get('route') or [c])[-1]['target_world'], float)[:2]


def _progress(c):
    return c['role'] in PROGRESS or '_entry_via_' in c['candidate_id']


def _observe_only(c):
    return c['role'] in ('OBSERVE', 'BOOTSTRAP') and not _progress(c)


def select_candidates(candidates, return_allowed, position=(0., 0., 0.), max_moves=MAX_MOVES, gains=None):
    """Drop what is not a choice; task order is left to the model.

    Removed: a return the task does not allow yet, aisle changes/inspections
    without new coverage while another option adds some (``gains``: new
    coverage per id, None to skip), a second observation move ending at the
    same place, observation-only moves while a move that makes progress exists
    (run 65aab932: two entry observations instead of inspecting), and moves
    beyond ``max_moves`` (longest routes first). Returns the kept
    candidates in input order and {candidate_id: reason} for the rest.
    """
    dropped = {}
    usable = []
    for c in candidates:
        if _returning(c) and not return_allowed:
            dropped[c['candidate_id']] = 'RETURN_NOT_AUTHORIZED'
        else:
            usable.append(c)
    if gains is not None and any((gains.get(c['candidate_id']) or 0.) > MIN_GAIN for c in usable):
        for c in list(usable):
            gain = gains.get(c['candidate_id'])
            if c['role'] in ('TRANSIT', 'INSPECT') and gain is not None and gain <= MIN_GAIN:
                dropped[c['candidate_id']] = 'NO_NEW_COVERAGE'
                usable.remove(c)
    progress = any(_progress(c) for c in usable)
    kept = []
    for c in sorted(usable, key=lambda c: c['candidate_id']):
        cid = c['candidate_id']
        if progress and _observe_only(c):
            dropped[cid] = 'OBSERVE_WHILE_PROGRESS_EXISTS'
            continue
        # Only local observation moves merge: two returns home may take different routes.
        twin = next((k for k in kept if c['role'] in ('OBSERVE', 'BOOTSTRAP') and k['role'] == c['role']
                     and np.linalg.norm(_end(k)-_end(c)) < SAME_PLACE), None)
        if twin is not None:
            dropped[cid] = 'SAME_PLACE_AS:'+twin['candidate_id']
            continue
        kept.append(c)
    moves = [c for c in kept if c['role'] not in ('HOLD', 'STOP')]
    if len(moves) > max_moves:
        def length(c):
            route = [np.asarray(position, float)]+[np.asarray(q['target_world'], float)
                                                    for q in (c.get('route') or [c])]
            return sum(np.linalg.norm(b[:2]-a[:2]) for a, b in zip(route, route[1:]))
        ranked = sorted(moves, key=lambda c: (c['role'] not in PROGRESS, length(c)))
        for c in ranked[max_moves:]:
            dropped[c['candidate_id']] = 'MOVE_CAP'
    ids = {c['candidate_id'] for c in kept}-set(dropped)
    return [c for c in candidates if c['candidate_id'] in ids], dropped


def assign_labels(candidates):
    """Letters in the given (already shuffled) order; HOLD/STOP is last."""
    ordered = ([c for c in candidates if c['role'] not in ('HOLD', 'STOP')]
               + [c for c in candidates if c['role'] in ('HOLD', 'STOP')])
    if len(ordered) > len(string.ascii_uppercase):
        raise ValueError('too many candidates for single-letter labels')
    return {c['candidate_id']: letter for c, letter in zip(ordered, string.ascii_uppercase)}


def bearing(vector):
    """Eight-way direction in the world-aligned image frame."""
    x, y = float(vector[0]), float(vector[1])
    if np.hypot(x, y) < 1e-6:
        return '原地'
    names = ['右', '右上', '上', '左上', '左', '左下', '下', '右下']
    return names[int(np.round(np.arctan2(y, x)/(np.pi/4))) % 8]


def _ends(corridor):
    d = np.asarray(corridor['direction'], float)
    return bearing(-d)+'端', bearing(d)+'端'


def aisle_facts(k, state, topology):
    """Observed facts about aisle ``k`` relative to the task; no recommendation."""
    by_index = {t['lateral_index']: t for t in topology.values()}
    uncovered = {row['corridor_id']: row['uncovered_length_m'] for row in state['coverage']['corridors']}
    current = (topology.get(state.get('current_corridor')) or {}).get('lateral_index')
    spec = state.get('task_spec') or {}
    facts = []
    started = any((uncovered.get(t['corridor_id']) or 0.) < .5 for t in by_index.values())
    # Outermost matters only for where S starts; later it pulled changes to the
    # far edge (decision test set: 8 of 10 errors picked the outermost aisle).
    if not started and (k == 1 or (k == max(by_index) and by_index[k].get('observed_edge'))):
        facts.append('最外侧')
    if current is not None and abs(k-current) == 1:
        facts.append('与当前通道相邻')
    if (uncovered.get(by_index[k]['corridor_id']) or 0.) < .5:
        facts.append('已巡完')
    if spec.get('corridors') and k not in spec['corridors']:
        facts.append('不在要求的通道内')
    if spec.get('target_lateral_index') == k:
        facts.append('指令的目标通道')
    return facts


def describe(c, public, topology, position, state=None):
    """One line per candidate from its public metrics; no coordinates.

    With ``state``, facts about the target aisle (outermost, adjacent, finished,
    outside the requested subset, instructed target) are appended.
    """
    text = _describe(c, public, topology, position)
    if state is not None and c['role'] not in ('HOLD', 'STOP', 'RETURN'):
        from .decision_bench import target_aisle
        k = target_aisle(c, topology)
        facts = aisle_facts(k, state, topology) if k in {t['lateral_index'] for t in topology.values()} else []
        if facts:
            text += '（第'+str(k)+'条：'+'，'.join(facts)+'）'
    return text


def _describe(c, public, topology, position):
    role, cid = c['role'], c['candidate_id']
    # Travel to the route end; the inspection itself is reported as new coverage.
    length = public.get('explicit_route_distance_m') or 0.
    gain = public.get('new_coverage_m')
    if gain is None:
        gain = public.get('corridor_uncovered_m')
    target = np.asarray(c['target_world'], float)[:2]-np.asarray(position, float)[:2]
    if role == 'HOLD':
        return '原地悬停，等待新观测'
    if role == 'STOP':
        return '停稳，结束任务'
    if role == 'RETURN':
        return f'返回起点，路程约{length:.0f}m'
    if role == 'REJOIN':
        return f'回到原通道继续，路程约{length:.0f}m'
    match = ROUTE_IDS.match(cid)
    corridor = topology.get(public.get('corridor_id'))
    if corridor is None:
        # Entry and end-observation moves name their aisle only in the id.
        named = match.group('cid') if match else cid.split('_entry_observe_')[0]
        corridor = topology.get(named)
    if (c.get('commitment') or {}).get('kind') == 'outer_observation':
        side = (c['commitment'].get('side') or '')
        normal = np.asarray(corridor['normal'], float) if corridor else np.array([0., 1.])
        return f'查看{bearing(-normal if side == "low" else normal)}侧最外一排外面是否还有通道，路程约{length:.0f}m'
    if corridor is None:
        return f'向{bearing(target)}移动{np.linalg.norm(target):.1f}m观察'
    k = corridor['lateral_index']
    low, high = _ends(corridor)
    side = (low if match and match.group('side') == 'low' else high) if match else None
    toward = None
    if public.get('inspection_direction') in (-1, 1):
        d = np.asarray(corridor['direction'], float)
        toward = bearing(public['inspection_direction']*d)
    if role == 'TRANSIT':
        skipped = public.get('skipped_lateral_indices') or []
        text = f'经{side}行末换到第{k}条，向{toward}巡检；新增{gain or 0:.1f}m，路程约{length:.0f}m'
        return text+(f'，跨过第{"、".join(map(str, skipped))}条' if skipped else '')
    if role == 'INSPECT':
        kind = '补巡' if '_gap_repair_' in cid else '巡检'
        return f'沿第{k}条向{toward}{kind}；新增{gain or 0:.1f}m'
    if match and match.group('kind') == 'entry_via':
        return f'绕过{side}行末外侧进入第{k}条，路程约{length:.0f}m'
    if match and match.group('kind') == 'connection_observe':
        return f'查看第{k}条{side}的连接，移动{np.linalg.norm(target):.1f}m'
    if match and match.group('kind') == 'observe':
        return f'查看第{k}条{side}行末，移动{np.linalg.norm(target):.1f}m'
    if '_entry_observe_' in cid:
        return f'靠近第{k}条入口观察，移动{np.linalg.norm(target):.1f}m'
    return f'向{bearing(target)}移动{np.linalg.norm(target):.1f}m观察'


def task_line(state):
    spec = state.get('task_spec') or {}
    parts = [INTENTS.get(state.get('intent'), str(state.get('intent')))]
    if state.get('intent') == 'COVER':
        # "The outermost" named both edges, and nothing said to finish the aisle
        # the vehicle is in: at aisle 1's end Qwen changed to aisle 2 or 3 in
        # every candidate order (runs 853c339a, b1accde0, 0ef13930).
        parts.append({'S': 'S形（从起点一侧的第1条开始；所在通道没巡完先沿它巡完，'
                           '再经行末换到相邻未巡的一条，反向巡检）',
                      'free': '任意顺序'}.get(spec.get('order'), str(spec.get('order'))))
        parts.append(f'只巡第{"、".join(map(str, spec["corridors"]))}条' if spec.get('corridors') else '全部通道')
    if spec.get('target_lateral_index'):
        parts.append(f'目标第{spec["target_lateral_index"]}条')
    if state.get('resume_after'):
        parts.append('完成后回到原任务')
    return '解析：'+'，'.join(parts)+'。'


def point_choice_prompt(state, candidates, labels, public, facts=False):
    """Short text for the two images; ``public`` maps candidate_id -> metrics."""
    position = state['position_world']
    topology = {t['corridor_id']: t for t in state.get('topology', [])}
    lines = [f'指令：{state["command"]}', task_line(state)]
    here = topology.get(state.get('current_corridor'))
    if here is None:
        lines.append('现在：不在任何通道内。')
    else:
        low, high = _ends(here)
        end = state.get('current_end')
        where = f'第{here["lateral_index"]}条' + ((low if end == 'low' else high) if end else '通道内')
        lines.append(f'现在：在{where}' + ('，行末已确认。' if state.get('at_confirmed_end') else '。'))
    previous = state.get('previous_intention') or {}
    if previous.get('corridor_id') in topology and previous.get('direction') in (-1, 1):
        t = topology[previous['corridor_id']]
        lines.append(f'上一段：沿第{t["lateral_index"]}条向{bearing(previous["direction"]*np.asarray(t["direction"]))}巡检。')
    if state.get('blocked_goal'):
        lines.append('上一个目标受阻未到达。')
    uncovered = {row['corridor_id']: row['uncovered_length_m'] for row in state['coverage']['corridors']}
    rows = []
    for t in sorted(state.get('topology', []), key=lambda t: t.get('lateral_index') or 0):
        low, high = _ends(t)
        u = uncovered.get(t['corridor_id']) or 0.
        text = f'{t["lateral_index"]} ' + ('已巡完' if u < .5 else f'未巡{u:.1f}m')
        open_ends = [name for name, side in ((low, 'low'), (high, 'high')) if not t['ends'][side]]
        if open_ends:
            text += '，'+'、'.join(open_ends)+'未确认'
        if t.get('observed_edge') and t.get('outer_open', True):
            text += '，外侧可能还有通道'
        rows.append(text)
    lines.append('通道：'+('｜'.join(rows) if rows else '尚未发现'))
    if state.get('completion', {}).get('eligible'):
        # Without the instruction's next step, a null aisle read as "do nothing"
        # and Qwen hovered after full coverage (runs c4294ad4, 4b639dae).
        returning = any(_returning(c) for c in candidates)
        lines.append('要求的通道已全部巡完' + ('，按指令返回起点。' if returning else '。'))
    lines.append('候选：')
    for c in sorted(candidates, key=lambda c: labels[c['candidate_id']]):
        lines.append(f'{labels[c["candidate_id"]]} '
                     f'{describe(c, public.get(c["candidate_id"], {}), topology, position, state if facts else None)}')
    rejection = state.get('last_rejection')
    if rejection:
        lines.append(f'上次选择未执行：{rejection.get("reason", rejection) if isinstance(rejection, dict) else rejection}')
    spec = state.get('task_spec') or {}
    whole_s = state.get('intent') == 'COVER' and spec.get('order') == 'S' and not spec.get('corridors')
    lines += [LEGEND, RULES.replace('规则：', '规则：'+S_RULE, 1) if whole_s else RULES]
    return '\n'.join(lines)


def candidate_orders(candidates, count):
    """``count`` rotations of the movement options (the first is the given order), HOLD/STOP last."""
    moves = [c for c in candidates if c['role'] not in ('HOLD', 'STOP')]
    rest = [c for c in candidates if c['role'] in ('HOLD', 'STOP')]
    count = max(1, min(count, len(moves)))
    return [moves[k:]+moves[:k]+rest for k in range(count)]


def combined_choice(answers, wait='HOLD'):
    """Candidate with the most letter probability summed over candidate orders.

    ``answers`` pairs each order's ``labels`` (candidate -> letter) with its
    reply: ``letter_probabilities`` (letter -> p), or without them the
    ``choice`` letter counted as probability 1. Each order is normalised over
    its offered letters, so a bias towards a letter position adds the same to
    every candidate across rotations. Qwen picked A in 21 of its 33 wrong
    answers on the decision test set; three rotations took aisle changes
    from 29/33 to 33/33 there. Ties go to the earlier candidate of the first
    order. ``wait`` (HOLD, always last) wins only when every order answered
    it: one order's 0.88 for HOLD outweighed a move both orders supported,
    and the unchanged view repeated that hover until the run timed out (run
    8388ee83). Returns (candidate_id or None, {candidate_id: score}).
    """
    totals = {}
    for labels, reply in answers:
        letters = reply.get('letter_probabilities')
        if letters is None:
            choice = (reply.get('choice') or {}).get('choice')
            letters = {choice: 1.} if choice else {}
        offered = {cid: float(letters.get(letter, 0.)) for cid, letter in labels.items()}
        mass = sum(offered.values())
        for cid, p in offered.items():
            totals[cid] = totals.get(cid, 0.)+(p/mass if mass > 0 else 0.)
    if not any(totals.values()):
        return None, totals
    best = max(totals, key=totals.get)
    if best == wait and any((reply.get('choice') or {}).get('choice') != labels.get(wait)
                            for labels, reply in answers):
        moves = {cid: score for cid, score in totals.items() if cid != wait}
        if any(moves.values()):
            best = max(moves, key=moves.get)
    return best, totals


def decision_from_choice(choice, labels, candidates):
    """Map a letter answer to the internal {candidate_id, behavior} decision."""
    by_label = {letter: cid for cid, letter in labels.items()}
    cid = by_label.get(choice)
    if cid is None:
        return None
    role = next(c['role'] for c in candidates if c['candidate_id'] == cid)
    return dict(candidate_id=cid, behavior=BEHAVIOR[role])
