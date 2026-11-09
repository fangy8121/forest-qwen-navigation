import json

import numpy as np
import pytest

from omni_drones.navigation.forest.qwen_subgoal import parse_coverage_command
from test_native_coverage_mission import finish, mission, snapshot


def command(**value):
    base = dict(intent='COVER', target=None, order=None, corridors=None, resume_after=False)
    return json.dumps(dict(base, **value))


@pytest.mark.parametrize('raw,intent,target', [
    (command(intent='VISIT_CORRIDOR', target=dict(relative='right', count=2), resume_after=True),
     'VISIT_CORRIDOR', dict(relative='right', count=2)),
    (command(intent='VISIT_CORRIDOR', target=dict(lateral_index=4)), 'VISIT_CORRIDOR', dict(lateral_index=4)),
    (command(order='free', corridors=[1, 3, 5]), 'COVER', None),
    (json.dumps(dict(intent='REVISIT_CORRIDOR', corridor_id='R01_R02', resume_after=True)),
     'REVISIT_CORRIDOR', dict(corridor_id='R01_R02')),
])
def test_v2_and_v1_commands_normalize_to_one_specification(raw, intent, target):
    parsed = parse_coverage_command(raw)
    assert parsed['intent'] == intent and parsed['target'] == target
    assert set(parsed) == {'intent', 'target', 'order', 'corridors', 'resume_after', 'corridor_id'}


@pytest.mark.parametrize('raw', [
    command(intent='VISIT_CORRIDOR'),
    command(intent='VISIT_CORRIDOR', target=dict(relative='up', count=1)),
    command(intent='VISIT_CORRIDOR', target=dict(lateral_index=0)),
    command(intent='REVISIT_CORRIDOR', target=dict(lateral_index=2)),
    command(corridors=[1, 1]),
    command(resume_after=True),
    json.dumps(dict(intent='VISIT_CORRIDOR', corridor_id='R01_R02', resume_after=False)),
])
def test_invalid_specifications_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_coverage_command(raw)


def test_relative_target_is_bound_to_the_heading_at_command_time(tmp_path):
    m = mission(tmp_path)
    try:
        # Corridors A, B, C at offsets 0, 5, 10 (normal +y); heading +x.
        m.latest_yaw = 0.
        assert m._relative_index(dict(relative='left', count=1)) == 3
        assert m._relative_index(dict(relative='right', count=1)) == 1
        assert m._relative_index(dict(relative='right', count=2)) is None
        m.latest_yaw = np.pi
        assert m._relative_index(dict(relative='left', count=1)) == 1
    finally:
        m.close()


def test_visit_may_leave_an_unfinished_aisle_and_only_moves_toward_target(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    try:
        m.intent = 'VISIT_CORRIDOR'
        m.task_spec = dict(order='S', corridors=None, target=dict(lateral_index=3), target_corridor=None)
        m._plan(p)
        ids = {c['candidate_id'] for c in m.choices}
        # B is unfinished, yet the instruction asks to go to C now.
        assert 'C_via_high' in ids and 'A_via_high' not in ids
        assert not any(c['role'] == 'INSPECT' and c['candidate_id'].startswith('B_') for c in m.choices)
        assert m.task_spec['target_corridor'] == 'C'
    finally:
        m.close()


def test_visit_completion_restores_the_interrupted_task(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 10., 4.])
    try:
        previous = m._context()
        m.intent, m.resume_after, m.resume_context = 'VISIT_CORRIDOR', True, previous
        m.task_spec = dict(order='S', corridors=None, target=dict(lateral_index=3), target_corridor=None)
        m.active_corridor = 'C'
        finish(m, 'C')
        m._plan(p)
        assert m.intent == 'COVER' and m.task_spec['target'] is None
        visit = next(e for e in m.events if e['kind'] == 'visit_completed')
        assert visit['corridor_id'] == 'C' and visit['lateral_index'] == 3
        assert m.events[-1]['kind'] == 'interrupted_task_restored' and m.events[-1]['restored_intent'] == 'COVER'
    finally:
        m.close()


def test_subset_completes_without_unrequested_aisles(tmp_path):
    m = mission(tmp_path)
    try:
        m.task_spec = dict(order='free', corridors=[1, 3], target=None, target_corridor=None)
        structure = m.structure.snapshot()
        assert not m._completion(structure)['eligible']
        finish(m, 'A'); finish(m, 'C')
        evidence = m._completion(structure)
        assert evidence['eligible'] and evidence['coverage_scope'] == 'REQUESTED_CORRIDOR_SUBSET'
        m.task_spec['corridors'] = [1, 4]
        assert 'LATERAL_4:NOT_OBSERVED' in m._completion(structure)['unresolved']
    finally:
        m.close()


def test_free_order_may_leave_an_unfinished_aisle_for_an_uncovered_one(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    try:
        m.task_spec = dict(order='free', corridors=None, target=None, target_corridor=None)
        m._plan(p)
        ids = {c['candidate_id'] for c in m.choices}
        assert {'B_positive', 'A_via_high', 'C_via_high'} <= ids
        m.task_spec['order'] = 'S'
        m._plan(p)
        assert not any('_via_' in c['candidate_id'] for c in m.choices)
    finally:
        m.close()


def test_visit_phase_text_names_target_and_resume(tmp_path):
    from omni_drones.navigation.forest.coverage_candidates import discovery_phase_context
    state = dict(intent='VISIT_CORRIDOR', resume_after=True, current_corridor='B',
        current_corridor_lateral_index=2, observed_corridor_count=3,
        task_spec=dict(target=dict(lateral_index=3), target_corridor='C', target_lateral_index=3),
        candidates=[dict(candidate_id='C_via_high', allowed_behavior='CHANGE_AISLE', new_coverage_m=20.,
                         corridor_lateral_index=3)])
    phase, text = discovery_phase_context(state)
    assert phase == 'VISIT' and '横向序号3的通道C' in text and '回到原任务断点' in text


def test_required_terminal_stop_does_not_wait_for_the_model(tmp_path):
    m = mission(tmp_path)
    try:
        for corridor_id in 'ABC':
            finish(m, corridor_id)
        m.structure.data['outer_sides'] = dict(low=dict(confirmed=True), high=dict(confirmed=True))
        m.intent, m.coverage_result = 'FINISH', 'COMPLETE_UNDER_ASSUMPTIONS'
        assert m.allows_stop()
        m.status = 'WAITING_NEXT'
        m.advance(5., np.asarray(m.home), np.zeros(3), 0., reference_settled=True)
        assert m.status == 'COMPLETE' and m.stop_requested
        assert m.events[-1]['source'] == 'task_spec_terminal_stop'
    finally:
        m.close()


def test_cover_only_fields_are_dropped_outside_cover():
    parsed = parse_coverage_command(command(intent='RESUME', order='S', corridors=[1]))
    assert parsed['intent'] == 'RESUME' and parsed['order'] is None and parsed['corridors'] is None


def test_malformed_command_is_retried_then_rejected_without_failing(tmp_path):
    m = mission(tmp_path)
    try:
        m.command_ready, m.command_pending = False, dict(request_id='command_1_0', task_revision=m.task_revision)
        error = dict(request_id='command_1_0', error='MODEL_OUTPUT: bad')
        m.worker.poll = lambda: error
        assert m._command_tick(1.) is False and m.status != 'FAILED'
        assert m.events[-1]['kind'] == 'command_retry'
        m.command_pending = dict(request_id='command_1_0', task_revision=m.task_revision)
        assert m._command_tick(2.) is True and m.status != 'FAILED'
        assert m.events[-1]['kind'] == 'command_rejected' and m.intent == 'COVER'
    finally:
        m.close()


def four_corridors(m):
    from test_native_coverage_mission import corridor
    m.structure.data['corridors'].append(corridor('D', 15.))


def test_headland_transfer_skips_unrequested_aisles(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    try:
        four_corridors(m)
        finish(m, 'B')
        # Subset: aisles 2 (B) and 4 (D); C (3) is only in the way.
        m.task_spec = dict(order='free', corridors=[2, 4], target=None, target_corridor=None)
        m._plan(p)
        transfer = next(c for c in m.choices if c['candidate_id'] == 'D_via_high')
        assert transfer['skipped_corridors'] == ['C'] and 'C_via_high' not in {c['candidate_id'] for c in m.choices}
        assert {tuple(e) for e in transfer['confirmed_ends']} == {('B', 'high'), ('C', 'high'), ('D', 'high')}
        xy = np.asarray([point['target_world'][:2] for point in transfer['route']])
        assert xy[0, 0] >= 22.2-1e-9 and xy[1, 0] >= 22.2-1e-9
    finally:
        m.close()


def test_s_order_keeps_adjacent_change_but_jumps_over_finished_aisles(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    try:
        four_corridors(m)
        finish(m, 'B')
        m._plan(p)
        ids = {c['candidate_id'] for c in m.choices}
        assert {'A_via_high', 'C_via_high'} <= ids and 'D_via_high' not in ids
        finish(m, 'A'); finish(m, 'C')
        m._plan(p)
        ids = {c['candidate_id'] for c in m.choices if '_via_' in c['candidate_id']}
        assert ids == {'D_via_high'}
    finally:
        m.close()


@pytest.mark.parametrize('second,mode', [([22., 5.], 'through'), ([18., 9.], 'stop')])
def test_interior_return_points_are_passed_unless_the_route_turns_sharply(tmp_path, second, mode):
    m = mission(tmp_path)
    p = np.array([14., 5., 3.9])
    try:
        route = [m._anchor(np.array([18., 5.]), 'RETURN'), m._anchor(np.array(second), 'RETURN')]
        selected = dict(route[0], candidate_id='RETURN_HOME', route=route)
        goal = m._native_goal(dict(candidate_id='RETURN_HOME', behavior='RETURN'), selected, 'decision_000',
            1., p, np.zeros(3), 0., snapshot(p))
        assert goal is not None and goal.completion_mode == mode
    finally:
        m.close()


def test_transfer_card_names_skipped_aisles_and_their_open_length(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    try:
        four_corridors(m)
        finish(m, 'B')
        m.task_spec = dict(order='free', corridors=None, target=None, target_corridor=None)
        m._plan(p)
        folder = tmp_path/'free'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        card = next(c for c in prepared['payload']['state']['candidates'] if c['candidate_id'] == 'D_via_high')
        assert card['skipped_lateral_indices'] == [3] and card['skipped_uncovered_m'] == 20.
        assert '跨过横向序号[3]（其中未巡20米' in prepared['payload']['prompt_phase_context']
    finally:
        m.close()
