"""Decision test set: teacher labels on synthetic scenes and relabelled images."""
import json

import pytest

from omni_drones.navigation.forest.decision_bench import build_item, permutations, synthetic_decision, teacher


def labelled(tmp_path, **kwargs):
    folder, context = synthetic_decision(tmp_path/'scene', **kwargs)
    payload = json.loads((folder/'input.json').read_text())
    public = {p['candidate_id']: p for p in payload['state']['candidates']}
    return folder, context, payload, teacher(payload['state'], payload['candidates'], public)


@pytest.mark.parametrize('kwargs, category, aisle', [
    (dict(), 'start', 'R01_R02'),
    (dict(covered=(1,), current=1, end='high', previous=1), 'change', 'R02_R03'),
    (dict(covered=(1, 2, 3), current=3, end='high', previous=1), 'change', 'R04_R05'),
    (dict(covered=(1,), current=1, end='high', previous=1, intent='VISIT_CORRIDOR', visit=4), 'visit', 'R04_R05'),
])
def test_teacher_accepts_only_the_task_order(tmp_path, kwargs, category, aisle):
    _, _, payload, (acceptable, found) = labelled(tmp_path, **kwargs)
    assert found == category and acceptable and all(cid.startswith(aisle) for cid in acceptable)
    moves = [c['candidate_id'] for c in payload['candidates'] if c['role'] not in ('HOLD', 'STOP')]
    assert len(moves) > len(acceptable)


def test_teacher_keeps_a_subset_and_accepts_any_return_after_coverage(tmp_path):
    _, _, _, (acceptable, category) = labelled(tmp_path/'a', covered=(1,), current=1, end='high', previous=1,
                                               subset=[1, 3, 5], order='free')
    assert category == 'subset' and {cid.split('_via_')[0] for cid in acceptable} == {'R03_R04', 'R05_R06'}
    _, _, _, (acceptable, category) = labelled(tmp_path/'b', covered=tuple(range(1, 7)), current=6, end='low',
                                               previous=-1)
    assert category == 'return' and set(acceptable) == {'RETURN_HOME', 'RETURN_HOME_HEADLAND'}


def test_items_relabel_both_images_for_each_candidate_order(tmp_path):
    folder, context, payload, _ = labelled(tmp_path, covered=(1,), current=1, end='high', previous=1)
    item, _ = build_item(folder, 'x', tmp_path/'item', context, count=3)
    assert len(item['variants']) == 3 and not item['trivial']
    assert len({tuple(v['order']) for v in item['variants']}) == 3
    for variant in item['variants']:
        assert all(path.endswith(('global.png', 'bev.png')) for path in variant['images'])
        assert max(variant['labels'].values()) == variant['labels']['HOLD']


def test_permutations_rotate_moves_and_keep_hold_last():
    candidates = [dict(candidate_id=c, role=r) for c, r in (('a', 'TRANSIT'), ('b', 'TRANSIT'), ('HOLD', 'HOLD'))]
    orders = [[c['candidate_id'] for c in p] for p in permutations(candidates, 3)]
    assert orders == [['a', 'b', 'HOLD'], ['b', 'a', 'HOLD']]


@pytest.mark.parametrize('kwargs', [dict(), dict(covered=(1,), current=1, end='high', previous=1)])
def test_teacher_worker_answers_a_live_decision_with_a_teacher_letter(tmp_path, kwargs):
    from omni_drones.navigation.forest.coverage_flight import TeacherWorker
    from omni_drones.navigation.forest.coverage_prompt import decision_from_choice
    folder, _, payload, (acceptable, category) = labelled(tmp_path, **kwargs)
    worker = TeacherWorker()
    assert worker.submit([str(folder/'global.png'), str(folder/'bev.png')], 'prompt', request_id='r',
                         response_schema='choice')
    reply = worker.poll()
    assert reply['request_id'] == 'r' and not worker.busy
    decision = decision_from_choice(reply['choice']['choice'], payload['labels'], payload['candidates'])
    assert decision['candidate_id'] in acceptable and reply['teacher']['category'] == category
    assert reply['teacher']['source'] == 'teacher' and json.loads(reply['raw']) == reply['choice']


@pytest.mark.parametrize('roles, source, expected', [
    (dict(a='BOOTSTRAP', b='BOOTSTRAP'), 'observe', 'b'),
    (dict(a='TRANSIT'), 'hold', 'HOLD'),
])
def test_teacher_worker_falls_back_to_observing_then_hold(tmp_path, monkeypatch, roles, source, expected):
    import omni_drones.navigation.forest.decision_bench as bench
    from omni_drones.navigation.forest.coverage_flight import TeacherWorker
    monkeypatch.setattr(bench, 'teacher', lambda *args: (None, 'no open aisle'))
    candidates = [dict(candidate_id=cid, role=role) for cid, role in roles.items()]
    candidates.append(dict(candidate_id='HOLD', role='HOLD'))
    labels = {c['candidate_id']: letter for c, letter in zip(candidates, 'ABC')}
    public = [dict(candidate_id='a', total_route_distance_m=.5), dict(candidate_id='b', total_route_distance_m=1.5)]
    (tmp_path/'input.json').write_text(json.dumps(dict(candidates=candidates, labels=labels, rule_reference=None,
                                                       state=dict(candidates=public, topology=[]))))
    worker = TeacherWorker()
    worker.submit([str(tmp_path/'global.png')], 'prompt', request_id='r', response_schema='choice')
    reply = worker.poll()
    assert reply['teacher']['source'] == source and reply['choice']['choice'] == labels[expected]


def test_teacher_worker_binds_the_whole_forest_s_instruction_as_cover():
    from omni_drones.navigation.forest.coverage_flight import TeacherWorker
    worker = TeacherWorker()
    text = '沿S形逐条巡检整个树林，在已确认的行末进入相邻未巡通道反向巡检，补齐遗漏，完成后返回起点并停稳。'
    worker.submit(None, json.dumps(dict(command=text, last_departed_corridor=None), ensure_ascii=False),
                  request_id='c', response_schema='coverage_command')
    command = worker.poll()['coverage_command']
    assert command['intent'] == 'COVER' and command['order'] == 'S' and command['corridors'] is None
