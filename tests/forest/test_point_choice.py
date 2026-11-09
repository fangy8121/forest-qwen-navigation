"""point_choice_v1: lettered candidates on two images, no task-order pruning."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import time

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from omni_drones.navigation.forest.coverage_flight import RuleWorker
from omni_drones.navigation.forest.coverage_mission import CoverageMission
from omni_drones.navigation.forest.coverage_prompt import (assign_labels, bearing, decision_from_choice,
    point_choice_prompt, select_candidates)
from omni_drones.navigation.forest.qwen_subgoal import parse_choice


def corridor(identifier, offset):
    return dict(id=identifier, direction=[1., 0.], normal=[0., 1.], offset_m=offset,
        span=[0., 20.], bordering_rows=[identifier+'a', identifier+'b'],
        ends={side: dict(s=s, confirmed=True) for side, s in [('low', 0.), ('high', 20.)]})


class Structure:
    last_error = None

    def __init__(self, data):
        self.data = data

    def snapshot(self):
        return deepcopy(self.data)


def structure(overview=False):
    x, y = np.meshgrid(np.arange(-5., 27.), np.arange(-6., 17.))
    vertices = np.c_[x.ravel(), y.ravel(), .1*x.ravel()]
    value = dict(corridors=[corridor('A', 0.), corridor('B', 5.), corridor('C', 10.)],
        rows=[], outer_sides={}, stem_candidates=[dict(xy=[x, y], radius_m=.2)
            for x in range(0, 21, 4) for y in (-2.5, 2.5, 7.5, 12.5)],
        terrain=dict(vertices=vertices.tolist(), timestamps=[1.]*len(vertices)))
    if overview:
        flags = np.zeros((90, 70), np.uint8)
        flags[10:70, 10:60] = 1
        flags[20:60:10, 15:55:12] |= 2
        value.update(public_work_area=dict(xy_bounds=[-4., 28., -6., 18.]),
            overview=dict(origin=[-8., -10.], resolution=.4, flags=flags))
    return value


def mission(tmp_path, decision_input='point_choice', decision_facts=False, decision_images='both',
            decision_orders=1, **kwargs):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2]/'cfg/task/ForestSurvey.yaml')
    result = CoverageMission(RuleWorker(), '沿S形巡检整个树林，然后返回起点并停稳。',
        [0., 5., 2.5], 0., SimpleNamespace(**dict(cfg.flight)), tmp_path,
        [-20., 40., -20., 30.], [-10., 30.], Structure(structure(**kwargs)),
        decision_policy='qwen', desired_agl=2.5, max_decisions=100, decision_input=decision_input,
        decision_facts=decision_facts, decision_images=decision_images, decision_orders=decision_orders)
    result.active_corridor = 'B'
    result.status, result.command_ready = 'WAITING_NEXT', True
    return result


def snapshot(position):
    grid = np.stack(np.meshgrid(np.arange(-5., 5.01, .2), np.arange(-4., 4.01, .2), np.arange(-1., 1.01, .2),
                                indexing='ij'), -1).reshape(-1, 3)
    return dict(timestamp=1., captured=time.perf_counter(), points=np.array([[30., 30., 0.]]),
        evidence=np.asarray(position)+grid)


def test_unfinished_aisle_keeps_aisle_changes_and_logs_the_rule_order(tmp_path):
    # S order kept only the current aisle here; point_choice leaves that to Qwen.
    (tmp_path/'legacy').mkdir()
    (tmp_path/'choice').mkdir()
    legacy, choice = mission(tmp_path/'legacy', 'legacy'), mission(tmp_path/'choice')
    p = np.array([15., 5., 4.])
    try:
        legacy._plan(p)
        choice._plan(p)
        old = {c['candidate_id'] for c in legacy.choices}
        new = {c['candidate_id'] for c in choice.choices}
        assert not any('_via_' in i for i in old)
        assert {'A_via_high', 'C_via_high'} <= new and old <= new
        assert set(choice.rule_reference) == old and legacy.rule_reference is None
    finally:
        legacy.close()
        choice.close()


@pytest.mark.parametrize('overview', [False, True])
def test_prepared_input_has_two_bounded_images_a_short_prompt_and_letters(tmp_path, overview):
    m = mission(tmp_path, overview=overview)
    p = np.array([15., 5., 4.])
    folder = tmp_path/'decision_000'
    folder.mkdir()
    try:
        m._plan(p)
        prepared = m._prepare_native(snapshot(p), p, 0., 'decision_000', folder, None)
        assert prepared['response_schema'] == 'choice' and len(prepared['image']) == 2
        for path in prepared['image']:
            with Image.open(path) as image:
                assert image.width*image.height <= 262144
        labels = prepared['labels']
        assert labels['HOLD'] == max(labels.values())
        assert sorted(labels.values()) == [chr(65+i) for i in range(len(labels))]
        text = prepared['prompt']
        assert len(text) < 1500 and '候选：' in text and 'candidate_id' not in text
        for cid, letter in labels.items():
            assert f'\n{letter} ' in text
        assert '经右端行末换到第1条' in text and '经右端行末换到第3条' in text
        payload = json.loads((folder/'input.json').read_text())
        assert payload['protocol'] == 'point_choice_v1' and payload['rule_reference'] is not None
        assert (folder/'replay.npz').exists() and (folder/'structure.json').exists()
    finally:
        m.close()


def test_online_input_can_send_the_global_image_alone_with_aisle_facts(tmp_path):
    # The decision bench setting: facts appended, BEV drawn but not sent.
    m = mission(tmp_path, decision_facts=True, decision_images='global')
    p = np.array([15., 5., 4.])
    folder = tmp_path/'decision_000'
    folder.mkdir()
    try:
        m._plan(p)
        prepared = m._prepare_native(snapshot(p), p, 0., 'decision_000', folder, None)
        assert prepared['image'] == [str(folder/'global.png')] and (folder/'bev.png').exists()
        assert '（第1条：' in prepared['prompt'] and '（第3条：' in prepared['prompt']
        view = json.loads((folder/'input.json').read_text())['view']
        assert view['sent_images'] == ['global.png'] and view['prompt_facts'] is True
    finally:
        m.close()


@pytest.mark.parametrize('spec, expected', [(None, True), (dict(order='free', corridors=None), False),
                                            (dict(order='S', corridors=[1, 3]), False)])
def test_only_a_whole_forest_s_task_states_how_s_starts_and_finishes_the_current_aisle(tmp_path, spec, expected):
    # At aisle 1's end Qwen changed aisles in every candidate order (runs
    # 853c339a, b1accde0); for other orders or a subset the rule would be wrong.
    from omni_drones.navigation.forest.coverage_prompt import S_RULE
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    folder = tmp_path/'decision_000'
    folder.mkdir()
    try:
        m._plan(p)
        if spec is not None:
            m.task_spec = dict(m.task_spec or {}, **spec)
        text = m._prepare_native(snapshot(p), p, 0., 'decision_000', folder, None)['prompt']
        assert (S_RULE in text) is expected
        assert ('从起点一侧的第1条开始；所在通道没巡完先沿它巡完' in text) is (spec is None or spec['order'] == 'S')
    finally:
        m.close()


def test_rotated_orders_letter_the_same_candidates_with_their_own_images_and_text(tmp_path):
    m = mission(tmp_path, decision_facts=True, decision_images='global', decision_orders=3)
    p = np.array([15., 5., 4.])
    folder = tmp_path/'decision_000'
    folder.mkdir()
    try:
        m._plan(p)
        prepared = m._prepare_native(snapshot(p), p, 0., 'decision_000', folder, None)
        orders = prepared['orders']
        moves = [c['candidate_id'] for c in prepared['candidates'] if c['role'] not in ('HOLD', 'STOP')]
        assert len(moves) >= 3 and len(orders) == 3 and orders[0]['labels'] == prepared['labels']
        for k, order in enumerate(orders):
            assert set(order['labels']) == set(prepared['labels']) and order['labels']['HOLD'] == max(order['labels'].values())
            assert order['labels'][moves[k]] == 'A'
            assert order['image'] == [str(folder/('global.png' if k == 0 else f'global_o{k}.png'))]
            assert Path(order['image'][0]).exists()
        assert len({order['prompt'] for order in orders}) == 3
        saved = json.loads((folder/'input.json').read_text())['orders']
        assert [o['images'] for o in saved] == [['global.png'], ['global_o1.png'], ['global_o2.png']]
        m.decision_orders = 1
        (tmp_path/'decision_001').mkdir()
        single = m._prepare_native(snapshot(p), p, 0., 'decision_001', tmp_path/'decision_001', None)
        assert 'orders' not in single and 'orders' not in json.loads((tmp_path/'decision_001/input.json').read_text())
    finally:
        m.close()


@pytest.mark.parametrize('confirmed', [False, True])
def test_outer_aisles_say_more_may_lie_beyond_only_until_that_side_is_confirmed(tmp_path, confirmed):
    # Runs 4b639dae, c65a882d: with every aisle done and both outer sides
    # confirmed, the prompt still said so and Qwen hovered instead of returning.
    m = mission(tmp_path)
    m.structure.data.update(rows=[dict(id='Aa', offset_m=-2.5), dict(id='Cb', offset_m=12.5)],
        outer_sides={side: dict(confirmed=confirmed) for side in ('low', 'high')})
    p = np.array([15., 5., 4.])
    folder = tmp_path/'decision_000'
    folder.mkdir()
    try:
        m._plan(p)
        prepared = m._prepare_native(snapshot(p), p, 0., 'decision_000', folder, None)
        aisles = next(line for line in prepared['prompt'].splitlines() if line.startswith('通道：'))
        assert aisles.count('外侧可能还有通道') == (0 if confirmed else 2)
    finally:
        m.close()


def test_unknown_image_setting_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='decision_images'):
        mission(tmp_path, decision_images='bev')


def test_letter_answers_map_to_the_offered_candidate_and_its_behaviour():
    candidates = [dict(candidate_id='C_via_high', role='TRANSIT'), dict(candidate_id='B_positive', role='INSPECT'),
                  dict(candidate_id='HOLD', role='HOLD')]
    labels = assign_labels(candidates)
    assert labels == {'C_via_high': 'A', 'B_positive': 'B', 'HOLD': 'C'}
    assert decision_from_choice('A', labels, candidates) == dict(candidate_id='C_via_high', behavior='CHANGE_AISLE')
    assert decision_from_choice('C', labels, candidates) == dict(candidate_id='HOLD', behavior='OBSERVE')
    assert decision_from_choice('D', labels, candidates) is None


@pytest.mark.parametrize('raw, ok', [('{"next_aisle":2,"choice":"B"}', True),
    ('```json\n{"next_aisle": null, "choice": "A"}\n```', True), ('{"choice":"B"}', False),
    ('{"next_aisle":1,"choice":"b"}', False), ('{"next_aisle":1,"choice":"AB"}', False),
    ('{"next_aisle":0,"choice":"A"}', False), ('{"next_aisle":"x","choice":"A"}', False),
    ('{"next_aisle":"0","choice":"A"}', False), ('{"next_aisle":1,"choice":"A","why":"x"}', False)])
def test_choice_parser_accepts_an_aisle_and_exactly_one_uppercase_letter(raw, ok):
    if ok:
        assert set(parse_choice(raw)) == {'next_aisle', 'choice'}
    else:
        with pytest.raises(ValueError):
            parse_choice(raw)


@pytest.mark.parametrize('aisle, parsed', [('"null"', None), ('"NULL"', None), ('"3"', 3)])
def test_a_quoted_aisle_does_not_void_the_letter(aisle, parsed):
    # The aisle is only logged; run c65a882d lost five return choices to "null".
    assert parse_choice('{"next_aisle": %s, "choice": "B"}' % aisle) == dict(next_aisle=parsed, choice='B')


def point(cid, role, xy, **extra):
    return dict(candidate_id=cid, role=role, target_world=[*xy, 2.], **extra)


def test_selection_drops_unauthorized_return_twins_and_observation_beside_progress():
    candidates = [point('scan_forward_150', 'OBSERVE', (1.5, 0.)), point('A_entry_observe_150', 'OBSERVE', (0., 1.5)),
                  point('B_entry_observe_150', 'OBSERVE', (.2, 1.5)), point('C_entry_via_low', 'OBSERVE', (0., 9.)),
                  point('A_positive', 'INSPECT', (4., 0.)),
                  point('RETURN_HOME', 'RETURN', (-3., 0.), requested_intent='RETURN_HOME')]
    kept, dropped = select_candidates(candidates, return_allowed=False)
    assert [c['candidate_id'] for c in kept] == ['C_entry_via_low', 'A_positive']
    assert dropped == {'RETURN_HOME': 'RETURN_NOT_AUTHORIZED', 'scan_forward_150': 'OBSERVE_WHILE_PROGRESS_EXISTS',
                       'A_entry_observe_150': 'OBSERVE_WHILE_PROGRESS_EXISTS',
                       'B_entry_observe_150': 'OBSERVE_WHILE_PROGRESS_EXISTS'}
    # Without a move that makes progress, observation is the way forward; twins merge.
    kept, dropped = select_candidates(candidates[:3], return_allowed=False)
    assert [c['candidate_id'] for c in kept] == ['scan_forward_150', 'A_entry_observe_150']
    assert dropped == {'B_entry_observe_150': 'SAME_PLACE_AS:A_entry_observe_150'}
    kept, dropped = select_candidates([point(f'scan_{i}', 'OBSERVE', (i, 0.)) for i in range(1, 11)], False,
                                      (0., 0., 2.), max_moves=8)
    assert len(kept) == 8 and sorted(dropped) == ['scan_10', 'scan_9']


def test_bearing_uses_the_image_frame():
    assert [bearing(v) for v in ([1, 0], [0, 1], [-1, 0], [0, -1], [1, 1])] == ['右', '上', '左', '下', '右上']


def test_aisle_changes_without_new_coverage_go_while_another_option_adds_some():
    # Run 79f68b73: an offered change to a finished aisle was chosen and cost 140 s.
    candidates = [point('A_via_high', 'TRANSIT', (20., 0.)), point('C_via_high', 'TRANSIT', (20., 10.)),
                  point('B_negative', 'INSPECT', (11., 5.))]
    kept, dropped = select_candidates(candidates, False, (15., 5., 2.),
                                      gains={'A_via_high': 0., 'C_via_high': 20., 'B_negative': None})
    assert [c['candidate_id'] for c in kept] == ['C_via_high', 'B_negative']
    assert dropped == {'A_via_high': 'NO_NEW_COVERAGE'}
    # With nothing left to cover, nothing is dropped for lack of gain.
    kept, dropped = select_candidates(candidates[:1], False, gains={'A_via_high': 0.})
    assert [c['candidate_id'] for c in kept] == ['A_via_high'] and dropped == {}


def test_two_returns_home_by_different_routes_are_both_offered():
    # Run 79f68b73: the headland return was merged into the ~400 m history return.
    history = point('RETURN_HOME', 'RETURN', (0., 5.), requested_intent='RETURN_HOME',
                    route=[dict(target_world=[x, 5., 2.]) for x in (60., 30., 0.)])
    headland = point('RETURN_HOME_HEADLAND', 'RETURN', (0., 5.), requested_intent='RETURN_HOME',
                     route=[dict(target_world=[x, y, 2.]) for x, y in ((96., 30.), (96., 5.), (0., 5.))])
    kept, dropped = select_candidates([history, headland], True, (90., 30., 2.))
    assert {c['candidate_id'] for c in kept} == {'RETURN_HOME', 'RETURN_HOME_HEADLAND'} and dropped == {}


@pytest.mark.parametrize('offered', [True, False])
def test_full_coverage_names_the_return_only_when_one_is_offered(offered):
    # Runs c4294ad4, 4b639dae: after "已全部巡完" alone Qwen answered null and
    # HOLD every time; the test set's returns went from 4/24 to 24/24.
    returning = point('RETURN_HOME', 'RETURN', (0., 5.), requested_intent='RETURN_HOME')
    hold = point('HOLD', 'HOLD', (30., 5.))
    candidates = [returning, hold] if offered else [hold]
    state = dict(command='巡检', intent='COVER', position_world=[30., 5., 2.], topology=[],
                 coverage=dict(corridors=[]), completion=dict(eligible=True))
    prompt = point_choice_prompt(state, candidates, assign_labels(candidates),
                                 {'RETURN_HOME': dict(explicit_route_distance_m=28.)})
    done = next(line for line in prompt.splitlines() if line.startswith('要求的通道已全部巡完'))
    assert done == ('要求的通道已全部巡完，按指令返回起点。' if offered else '要求的通道已全部巡完。')
    state['completion']['eligible'] = False
    assert '全部巡完' not in point_choice_prompt(state, candidates, assign_labels(candidates),
                                                 {'RETURN_HOME': dict(explicit_route_distance_m=28.)})
