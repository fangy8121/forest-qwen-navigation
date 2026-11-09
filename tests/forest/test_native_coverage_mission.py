"""Native mission contracts using observed structure, without model/simulator."""
from copy import deepcopy
from types import SimpleNamespace
import json
import time

import numpy as np
import pytest

from omni_drones.navigation.forest.coverage_candidates import coverage_choices
from omni_drones.navigation.forest.coverage_candidates import (observed_task_memory,
    compact_prompt_state, discovery_phase_context, candidate_intent_metrics)
from omni_drones.navigation.forest.coverage_mission import CoverageMission
from omni_drones.navigation.forest.coverage_flight import RuleWorker


def corridor(identifier, offset, confirmed=True):
    return dict(id=identifier, direction=[1., 0.], normal=[0., 1.], offset_m=offset,
        span=[0., 20.], bordering_rows=[identifier+'a', identifier+'b'],
        ends={side:dict(s=s, confirmed=confirmed) for side, s in [('low', 0.), ('high', 20.)]})


def structure(confirmed=True):
    x, y = np.meshgrid(np.arange(-5., 27.), np.arange(-6., 17.))
    vertices = np.c_[x.ravel(), y.ravel(), .1*x.ravel()]
    return dict(corridors=[corridor('A', 0., confirmed), corridor('B', 5., confirmed),
        corridor('C', 10., confirmed)], rows=[], outer_sides={},
        terrain=dict(vertices=vertices.tolist(), timestamps=[1.]*len(vertices)))


class Structure:
    last_error = None

    def __init__(self, data):
        self.data = data

    def snapshot(self):
        return deepcopy(self.data)


def finish(m, corridor='B'):
    # Cross-row options exist only once the current aisle is fully covered.
    m.ledger.covered[corridor], m.ledger.directions[corridor] = [[0., 20.]], [1., 0.]


def mission(tmp_path, *, agl=2.5, confirmed=True):
    from omegaconf import OmegaConf
    from pathlib import Path
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2]/'cfg/task/ForestSurvey.yaml')
    result = CoverageMission(RuleWorker(), '沿S形巡检整个树林，然后返回起点并停稳。',
        [0., 5., 2.5], 0., SimpleNamespace(**dict(cfg.flight)), tmp_path,
        [-20., 40., -20., 30.], [-10., 30.], Structure(structure(confirmed)),
        decision_policy='qwen', desired_agl=agl, max_decisions=100)
    result.active_corridor = 'B'
    result.status, result.command_ready = 'WAITING_NEXT', True
    result.latest = snapshot(result.home)
    return result


def snapshot(position):
    return dict(timestamp=1., captured=time.perf_counter(), points=np.array([[30., 30., 0.]]),
        evidence=np.asarray(position)[None])


def test_qwen_receives_both_directions_and_both_adjacent_corridors(tmp_path):
    m = mission(tmp_path)
    try:
        for x, side, far in ((5., 'low', 'high'), (15., 'high', 'low')):
            m.ledger.covered.pop('B', None)
            m._plan(np.array([x, 5., 2.5+.1*x]))
            assert not m.route and m.commitment is None
            ids = {c['candidate_id'] for c in m.choices}
            # Unfinished current aisle: both directions, no aisle change yet.
            assert {'B_positive', 'B_negative'} <= ids and not any('_via_' in i for i in ids)
            finish(m)
            m._plan(np.array([x, 5., 2.5+.1*x]))
            ids = {c['candidate_id'] for c in m.choices}
            assert {'A_via_'+side, 'C_via_'+side} <= ids
            assert 'A_via_'+far not in ids and 'C_via_'+far not in ids
    finally:
        m.close()


def test_unconfirmed_ends_do_not_authorize_cross_row_route(tmp_path):
    m = mission(tmp_path, confirmed=False)
    try:
        m._plan(np.array([10., 5., 3.5]))
        assert not any('_via_' in c['candidate_id'] for c in m.choices)
        assert {'B_observe_low', 'B_observe_high'} <= {c['candidate_id'] for c in m.choices}
    finally:
        m.close()


def test_selected_neighbor_is_bound_without_rule_replacement(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    try:
        finish(m)
        m._plan(p)
        selected = next(c for c in m.choices if c['candidate_id'] == 'C_via_high')
        decision = dict(candidate_id=selected['candidate_id'], behavior='CHANGE_AISLE')
        pending = dict(task_revision=1, goal_revision=1, request_id='decision_000')
        goal = m._activate(decision, selected, pending, 1., p, np.zeros(3), 0., snapshot(p))
        assert goal is not None
        assert m.commitment['corridor_id'] == 'C' and m.commitment['direction'] == -1
        assert m.route[-1]['corridor_id'] == 'C'
        assert m.events[-1]['request_id'] == 'decision_000'
        assert m.events[-1]['target_world'] == selected['target_world']
    finally:
        m.close()


def test_ground_lifting_changes_height_and_rejects_unknown_support(tmp_path):
    m = mission(tmp_path)
    try:
        assert np.isclose(m._anchor([10., 5.], 'INSPECT', 'B')['target_world'][2], 3.5)
        with pytest.raises(ValueError, match='GROUND'):
            m._anchor([100., 5.], 'INSPECT', 'B')
        points = m._lift_guidance([0., 5., 2.5], [10., 5., 3.5])
        assert np.max(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)) <= .5+1e-10
        assert np.allclose(points[:, 2], .1*points[:, 0]+2.5)
        with pytest.raises(ValueError, match='GROUND'):
            m._lift_guidance([0., 5., 2.5], [35., 5., 6.])
    finally:
        m.close()


def test_warmup_hold_needs_no_terrain(tmp_path):
    m = mission(tmp_path)
    try:
        m.structure.data['terrain'] = {}
        assert np.array_equal(m.guidance_points(m.home), m.home[None])
    finally:
        m.close()


def test_committed_interior_extension_uses_through_gate_and_original_request(tmp_path):
    m = mission(tmp_path)
    p = np.array([1., 5., 2.6])
    try:
        m._plan(p)
        selected = next(c for c in m.choices if c['candidate_id'] == 'B_positive')
        decision = dict(candidate_id='B_positive', behavior='FOLLOW')
        goal = m._activate(decision, selected,
            dict(task_revision=1, goal_revision=1, request_id='decision_000'),
            1., p, np.zeros(3), 0., snapshot(p))
        assert goal.completion_mode == 'through'
        assert m.continuation[0] > goal.target_world[0]
        m.route = []
        m.status = 'WAITING_NEXT'
        p = np.asarray(goal.target_world)+[.01, 0., .001]
        m._plan(p)
        selected = dict(m.route[0], candidate_id='continue_3')
        following = m._native_goal(dict(candidate_id='continue_3', behavior='FOLLOW'), selected,
            m.route_request_id, 2., p, np.array([.4, 0., .04]), 0., snapshot(p), continuation=True)
        assert following.target_world[0] > goal.target_world[0]
        assert m.events[-1]['kind'] == 'coverage_continuation_activated'
        assert m.events[-1]['source_request_id'] == 'decision_000'
        assert m.events[-1]['parent_candidate_id'] == 'B_positive'
    finally:
        m.close()


def test_supported_height_does_not_hide_ground_gap_along_connection(tmp_path):
    m = mission(tmp_path)
    try:
        terrain = m.structure.data['terrain']
        vertices = np.asarray(terrain['vertices'])
        keep = (vertices[:, 0] < 4.) | (vertices[:, 0] > 7.)
        terrain['vertices'] = vertices[keep].tolist()
        terrain['timestamps'] = np.asarray(terrain['timestamps'])[keep].tolist()
        selected = m._anchor([10., 5.], 'INSPECT', 'B')
        valid, _, _ = m._validate(snapshot([0., 5., 2.5]), np.array([0., 5., 2.5]), 0., selected)
        assert not valid
    finally:
        m.close()


def test_input_records_all_alternatives_without_a_preselected_route(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    try:
        finish(m)
        m._plan(p)
        folder = tmp_path/'decision'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        payload = json.loads((folder/'input.json').read_text())
        # Finished aisle B: both adjacent aisle changes plus HOLD.
        assert len(prepared['candidates']) >= 3
        assert payload['movement_alternatives'] >= 2
        assert not m.route
        assert payload['state']['current_corridor'] == 'B'
        assert all('correct_next' not in c for c in payload['candidates'])
    finally:
        m.close()


def test_lateral_start_has_observation_recovery_for_observed_corridor(tmp_path):
    m = mission(tmp_path)
    try:
        m.active_corridor = None
        m._plan(np.array([-2., 2., 2.3]))
        recoveries = [c for c in m.choices if c.get('observation_slot', '').endswith('entry_observe')]
        assert len({c['observation_slot'] for c in recoveries}) == 3
        assert all(c['requires_observed_connection'] for c in recoveries)
        assert all(np.linalg.norm(np.asarray(c['target_world'])[:2]-[-2., 2.]) <= 1.5+1e-8
            for c in recoveries)
    finally:
        m.close()


def test_neighbor_end_can_be_observed_before_connection_is_confirmed(tmp_path):
    m = mission(tmp_path)
    try:
        m.structure.data['corridors'][2]['ends']['high']['confirmed'] = False
        m._plan(np.array([19., 5., 4.4]))
        candidate = next(c for c in m.choices if c['candidate_id'] == 'C_connection_observe_high')
        assert candidate['role'] == 'OBSERVE' and candidate['requires_observed_connection']
        assert not any(c['candidate_id'] == 'C_via_high' for c in m.choices)
    finally:
        m.close()


def test_lost_completion_evidence_revokes_finish_and_pending_stop(tmp_path):
    m = mission(tmp_path)
    try:
        m.coverage_result = 'COMPLETE_UNDER_ASSUMPTIONS'
        m.intent, m.stop_requested = 'FINISH', True
        m.pending = dict(task_revision=1)
        assert not m.allows_stop()
        m.update(1., m.home, np.zeros(3), 0.)
        assert m.intent == 'COVER' and m.status == 'WAITING_NEXT'
        assert not m.stop_requested and m.task_revision == 2 and m.switching
    finally:
        m.close()


def test_observation_hold_waits_without_burning_schema_rejection_budget(tmp_path):
    m = mission(tmp_path)
    try:
        for now in (1., 3., 5.):
            m._reject('OBSERVE_RETRY', now, 'waiting for observation')
        assert m.rejections == 0 and m.status == 'WAITING_NEXT'
        assert m.next_policy_time == 7.
        m._reject('OBSERVE_RETRY', 1.+m.no_motion_seconds, 'still waiting')
        assert m.status == 'FAILED' and m.reason == 'observation_no_progress'
    finally:
        m.close()


def test_early_uncredited_motion_has_explicit_gap_repair_candidate(tmp_path):
    m = mission(tmp_path)
    try:
        m.ledger.covered['B'] = [[4., 20.]]
        m.ledger.directions['B'] = [1., 0.]
        m._plan(np.array([10., 5., 3.5]))
        candidate = next(c for c in m.choices if c['candidate_id'] == 'B_gap_repair_0')
        assert candidate['commitment']['direction'] == -1
        assert candidate['commitment']['until_s'] == 0.
    finally:
        m.close()


def test_reopened_end_invalidates_every_committed_connection_stage(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    try:
        finish(m)
        m._plan(p)
        selected = next(c for c in m.choices if c['candidate_id'] == 'C_via_high')
        m._activate(dict(candidate_id=selected['candidate_id'], behavior='CHANGE_AISLE'), selected,
            dict(task_revision=1, goal_revision=1, request_id='decision_000'),
            1., p, np.zeros(3), 0., snapshot(p))
        assert all(anchor['confirmed_ends'] == selected['confirmed_ends'] for anchor in m.route)
        m.structure.data['corridors'][2]['ends']['high']['confirmed'] = False
        assert not m._validate(snapshot(p), p, 0., m.route[1])[0]
    finally:
        m.close()


def test_endpoint_support_does_not_advertise_unsupported_connection_to_model(tmp_path):
    m = mission(tmp_path)
    p = np.array([0., 5., 2.5])
    try:
        terrain = m.structure.data['terrain']
        vertices = np.asarray(terrain['vertices'])
        keep = (vertices[:, 0] < 4.) | (vertices[:, 0] > 7.)
        terrain['vertices'] = vertices[keep].tolist()
        terrain['timestamps'] = np.asarray(terrain['timestamps'])[keep].tolist()
        point = m._anchor([10., 5.], 'INSPECT', 'B')
        m.choices = [dict(point, candidate_id='B_positive', commitment=dict(corridor_id='B', direction=1))]
        folder = tmp_path/'diagnostic'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        assert [c['candidate_id'] for c in prepared['candidates']] == ['HOLD']
        assert prepared['observation_only']
        assert not prepared['payload']['model_requested']
        assert prepared['payload']['admission_reason'] == 'NO_SUPPORTED_MOVEMENT'
    finally:
        m.close()


def test_transient_geometry_wait_has_independent_budget(tmp_path):
    m = mission(tmp_path)
    try:
        m._reject('CANDIDATE_NO_LONGER_OBSERVED', 1.)
        m._reject('GROUND_SUPPORT', 3.)
        assert m.rejections == 0 and m.status == 'WAITING_NEXT'
        assert m.next_policy_time == 5.
        m._reject('GROUND_SUPPORT', 1.+m.no_motion_seconds)
        assert m.status == 'FAILED' and m.reason == 'observation_no_progress'
    finally:
        m.close()


def test_observation_only_preparation_never_submits_to_model(tmp_path):
    from concurrent.futures import Future
    m = mission(tmp_path)
    try:
        m.pending = dict(request_id='decision_000')
        m.preparation = Future()
        m.preparation.set_result(dict(observation_only=True, payload=dict(
            request_id='decision_000', admission_reason='NO_SUPPORTED_MOVEMENT')))
        assert m.advance(1., m.home, np.zeros(3), 0.) is None
        assert not m.worker.busy and m.pending is None and m.preparation is None
        assert m.rejections == 0 and m.next_policy_time == 3.
        assert any(e['kind'] == 'observation_input_saved' for e in m.events)
    finally:
        m.close()


def test_blocked_straight_entry_retains_observed_bent_approach(tmp_path):
    m = mission(tmp_path)
    try:
        m.active_corridor = None
        m.structure.data['corridors'] = [corridor('ENTRY', 0., confirmed=False)]
        m.structure.data['corridors'][0]['span'] = [2., 20.]
        vertices = np.asarray(m.structure.data['terrain']['vertices'])
        vertices[:, 2] = 0.
        m.structure.data['terrain']['vertices'] = vertices.tolist()
        x, y, z = np.meshgrid(np.arange(-2., 5.01, .2), np.arange(-3., 4.01, .2),
            np.arange(1.4, 3.61, .2), indexing='ij')
        evidence = np.c_[x.ravel(), y.ravel(), z.ravel()]

        def prepared_at(index, position):
            m._plan(position)
            s = dict(timestamp=1., captured=time.perf_counter(),
                points=np.array([[1.1, 0., 2.5]]), evidence=evidence)
            folder = tmp_path/f'curve_{index}'
            folder.mkdir()
            return m._prepare(s, position, 0., f'decision_{index:03d}', folder, [], None)

        first = prepared_at(0, np.array([0., 0., 2.5]))
        assert not any(c.get('observation_slot') == 'ENTRY_entry_observe' for c in first['candidates'])
        left = next(c for c in first['candidates'] if c.get('observation_slot') == 'scan_left')
        assert np.allclose(left['target_world'], [0., 1.5, 2.5])
        second = prepared_at(1, np.asarray(left['target_world']))
        forward = next(c for c in second['candidates'] if c.get('observation_slot') == 'scan_forward')
        assert np.allclose(forward['target_world'], [1.5, 1.5, 2.5])
        third = prepared_at(2, np.asarray(forward['target_world']))
        assert any(c.get('observation_slot') == 'ENTRY_entry_observe' for c in third['candidates'])
        # Every retained step used the same real body-clear/free-evidence checks;
        # a curved observation sequence exists without authorizing the blocked chord.
        assert any(not r['checks'].get('obstacle_clear', True) or not r['checks'].get('endpoint_clear', True)
            for r in first['payload']['rejected_geometry'])
    finally:
        m.close()


def activate_blocked_monitor(m, now, *, through=False):
    from omni_drones.navigation.forest.subgoal import GoalMonitor, Subgoal
    goal = Subgoal(f'blocked_{len(m.goals)}', len(m.goals)+1, (18., 5., 4.3),
        no_motion_seconds=.2, timeout_seconds=5.,
        completion_mode='through' if through else 'stop',
        approach_direction=(1., 0., 0.) if through else None,
        dwell_seconds=0. if through else .5)
    m.goals.append(goal)
    m.index = len(m.goals)-1
    m.monitor = GoalMonitor(goal, now)
    m.status = 'RUNNING'
    m.latest['timestamp'], m.latest['captured'] = float(now), time.perf_counter()
    m._selected = dict(candidate_id='B_positive', behavior='FOLLOW', role='INSPECT', corridor_id='B')
    m.commitment = dict(corridor_id='B', direction=1)
    m.route = [dict(target_world=list(goal.target_world), role='INSPECT', corridor_id='B')]


def test_local_execution_blockage_preserves_progress_and_requests_new_policy(tmp_path):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        m.ledger.covered['B'] = [[0., 4.]]
        activate_blocked_monitor(m, 0., through=True)
        m.update(0., p, np.zeros(3), 0.)
        m.update(.3, p, np.zeros(3), 0.)
        assert m.status == 'WAITING_NEXT' and m.requires_handoff_stop
        assert m.recoveries == 1 and not m.route and m.commitment is None
        assert m.results[-1]['status'] == 'FAILED' and m.results[-1]['reason'] == 'no_motion'
        assert m.ledger.covered['B'] == [[0., 4.]]
        assert not any(e['kind'] == 'task_failed' for e in m.events)
        assert m.advance(.31, p, np.array([.3, 0., 0.]), 0.) is None
        assert m.decisions == 0
        m.latest = snapshot(p)
        m.latest['timestamp'] = .4
        assert m.advance(.4, p, np.zeros(3), 0., reference_settled=True) is None
        assert m.decisions == 1 and m.preparation is not None
        prepared = m.preparation.result(timeout=5.)
        assert prepared['payload']['state']['blocked_goal']['reason'] == 'no_motion'
        assert prepared['payload']['state']['recoveries_used'] == 1
    finally:
        m.close()


@pytest.mark.parametrize('fault', ['STALE', 'uncertified_stop', 'timeout', 'no_motion'])
def test_incoming_execution_fault_stays_latched_and_cannot_use_recovery(tmp_path, fault):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        activate_blocked_monitor(m, 0.)
        m.update(0., p, np.zeros(3), 0.)
        m.update(.3, p, np.zeros(3), 0., failure=fault)
        assert m.status == 'FAILED' and m.reason == fault
        assert m.recoveries == 0
        m.update(.4, p, np.zeros(3), 0.)
        assert m.status == 'FAILED'
    finally:
        m.close()


def test_local_recovery_budget_is_bounded_across_goals(tmp_path):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        for index in range(4):
            activate_blocked_monitor(m, float(index))
            m.update(float(index), p, np.zeros(3), 0.)
            m.update(index+.3, p, np.zeros(3), 0.)
        assert m.recoveries == 3
        assert m.status == 'FAILED' and m.reason == 'local_recovery_budget:no_motion'
        assert sum(r['status'] == 'FAILED' for r in m.results) == 4
    finally:
        m.close()


def test_native_definition_identifies_real_qwen_coverage_protocol(tmp_path):
    m = mission(tmp_path)
    try:
        definition = m.definition()
        assert definition['protocol'] == 'qwen_native_forest_s_coverage_v1'
        assert definition['recovery_budget'] == 3
    finally:
        m.close()


def test_observed_memory_features_are_candidate_order_independent():
    s = dict(stem_candidates=[dict(xy=[4., -1.]), dict(xy=[4., 1.]), dict(xy=[6., 0.])])
    history = [dict(candidate_id=f'past_{i}', behavior='FOLLOW', actual_position=[float(i), 0., 2.5],
        actual_speed_mps=0., completed_at=float(i)) for i in range(12)]
    candidates = [dict(candidate_id='toward', target_world=[3., 0., 2.5]),
        dict(candidate_id='away', target_world=[-1., 0., 2.5]),
        dict(candidate_id='revisit', target_world=[8., 0., 2.5])]
    memory, features = observed_task_memory(s, [0., 0., 2.5], 0., history, candidates)
    reversed_memory, reversed_features = observed_task_memory(s, [0., 0., 2.5], 0., history, candidates[::-1])
    assert memory == reversed_memory and features == reversed_features
    assert [row['candidate_id'] for row in memory['recent_completed_actual_goals']] == [f'past_{i}' for i in range(4, 12)]
    assert memory['observed_structure']['count'] == 3
    assert memory['observed_structure']['centroid_bearing_body_deg'] == 0.
    assert features['toward']['median_structure_range_change_m'] < 0.
    assert features['away']['median_structure_range_change_m'] > 0.
    assert features['revisit']['nearest_previous_observation_m'] == 0.
    assert all(key not in memory for key in ('chosen_id', 'correct_next', 'recommended_candidate'))


def test_raw_structure_fallback_is_not_labeled_as_confirmed_trees():
    s = dict(stem_candidates=[], structure_points_xy=[[3., 0.], [5., 0.]])
    memory, features = observed_task_memory(s, [0., 0., 2.5], np.pi/2, [],
        [dict(candidate_id='test', target_world=[1., 0., 2.5])])
    observed = memory['observed_structure']
    assert observed['kind'] == 'observed_height_band_returns_not_confirmed_trees'
    assert memory['detected_stem_count'] == 0 and observed['count'] == 2
    assert np.allclose(observed['centroid_body_xy_m'], [0., -4.])
    assert observed['centroid_bearing_body_deg'] == -90.
    assert features['test']['median_structure_range_change_m'] == -1.


def test_native_input_includes_actual_history_and_geometry_for_every_candidate(tmp_path):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    history = [dict(candidate_id='past', behavior='FOLLOW', actual_position=[8., 5., 3.3],
        actual_speed_mps=0., completed_at=.2, teacher_route='must not enter input')]
    try:
        m.structure.data['stem_candidates'] = [dict(xy=[14., 3.]), dict(xy=[14., 7.])]
        m._plan(p)
        folder = tmp_path/'memory_input'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, history, None)
        state = prepared['payload']['state']
        assert state['observation_memory']['recent_completed_actual_goals'][0]['actual_position'] == [8., 5., 3.3]
        assert state['observation_memory']['observed_structure']['count'] == 2
        assert all('median_structure_range_change_m' in c['geometry'] for c in state['candidates'])
        assert 'teacher_route' not in json.dumps(state)
        assert not any(key in state for key in ('chosen_id', 'correct_next', 'recommended_candidate'))
    finally:
        m.close()


def test_bootstrap_budget_is_configurable_beyond_eight_observations(tmp_path):
    m = mission(tmp_path)
    try:
        m.active_corridor = None
        m.structure.data['corridors'] = []
        m.probes['bootstrap'] = 8
        m._plan(np.array([0., 5., 2.5]))
        assert any(c['role'] == 'BOOTSTRAP' for c in m.choices)
        m.max_bootstrap_observations = 8
        m._plan(np.array([0., 5., 2.5]))
        assert not any(c['role'] == 'BOOTSTRAP' for c in m.choices)
        assert m.definition()['max_bootstrap_observations'] == 8
    finally:
        m.close()


def test_bootstrap_direction_labels_follow_measured_body_yaw(tmp_path):
    m = mission(tmp_path)
    try:
        m.active_corridor = None
        m.structure.data['corridors'] = []
        m.latest_yaw = np.pi/2
        m._plan(np.array([0., 5., 2.5]))
        forward = next(c for c in m.choices if c['candidate_id'] == 'observe_forward_150')
        assert np.allclose(forward['target_world'][:2], [0., 6.5])
        right = next(c for c in m.choices if c['candidate_id'] == 'observe_right_150')
        assert np.allclose(right['target_world'][:2], [1.5, 5.])
    finally:
        m.close()


def return_candidate(m):
    point = dict(target_world=m.home.tolist(), role='RETURN', corridor_id=None)
    return dict(point, candidate_id='RETURN_HOME', id='RETURN_HOME', route=[point],
        requested_intent='RETURN_HOME', commitment=None)


def test_model_cannot_turn_incomplete_coverage_into_unrequested_return(tmp_path):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        selected = return_candidate(m)
        decision = dict(candidate_id='RETURN_HOME', behavior='RETURN')
        assert m._activate(decision, selected,
            dict(task_revision=1, goal_revision=1, request_id='decision_000'),
            1., p, np.zeros(3), 0., snapshot(p)) is None
        assert m.intent == 'COVER' and m.status == 'WAITING_NEXT'
        assert m.rejections == 1 and m.last_rejection['reason'] == 'EARLY_RETURN_UNFINISHED'
        assert decision == dict(candidate_id='RETURN_HOME', behavior='RETURN')
        assert len(m.goals) == 1
    finally:
        m.close()


@pytest.mark.parametrize('explicit_runtime_return', [False, True])
def test_completed_coverage_or_explicit_runtime_return_authorizes_home(tmp_path, explicit_runtime_return):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        if explicit_runtime_return:
            m.intent = 'RETURN_HOME'
        else:
            for c in m.structure.data['corridors']:
                m.ledger.covered[c['id']] = [[0., 20.]]
            m.structure.data['outer_sides'] = {side:dict(confirmed=True) for side in ('low', 'high')}
            m.coverage_result = 'COMPLETE_UNDER_ASSUMPTIONS'
        selected = return_candidate(m)
        goal = m._activate(dict(candidate_id='RETURN_HOME', behavior='RETURN'), selected,
            dict(task_revision=1, goal_revision=1, request_id='decision_000'),
            1., p, np.zeros(3), 0., snapshot(p))
        assert goal is not None and m.intent == 'RETURN_HOME' and m.rejections == 0
    finally:
        m.close()


def test_return_authorization_is_visible_without_hiding_the_candidate(tmp_path):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        m.choices = [return_candidate(m)]
        folder = tmp_path/'return_phase'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        public = next(c for c in prepared['payload']['state']['candidates'] if c['candidate_id'] == 'RETURN_HOME')
        assert public['authorized_after_coverage'] is False
        assert public['authorized_now'] is False
        assert any(c['candidate_id'] == 'RETURN_HOME' for c in prepared['candidates'])
        assert '暂时不能推进可选择HOLD和OBSERVE，不能把未完成任务改成返航结束。' in prepared['prompt']
    finally:
        m.close()


def unresolved_gap():
    return dict(id='gap_A_C', bordering_rows=['RA', 'RC'], direction=[1., 0.],
        normal=[0., 1.], lateral_span=[-5., 10.], span=[0., 20.],
        observation_target_xy=[10., 2.5], status='UNRESOLVED_MISSING_ROW_OR_OPEN_SPACE')


def test_wide_row_gap_exposes_frontier_and_generic_observations_without_aisle(tmp_path):
    m = mission(tmp_path)
    try:
        m.structure.data['corridors'] = []
        m.structure.data['unresolved_row_gaps'] = [unresolved_gap()]
        m.active_corridor = None
        m._plan(np.array([5., 0., 3.]))
        frontiers = [c for c in m.choices if c.get('frontier_id') == 'gap_A_C']
        assert frontiers and all(c['role'] == 'OBSERVE' and c['corridor_id'] is None for c in frontiers)
        assert all(c['requires_observed_connection'] for c in frontiers)
        assert len({c['observation_slot'] for c in m.choices if c.get('observation_slot', '').startswith('scan_')}) == 8
        assert not any(c.get('commitment') for c in m.choices)
    finally:
        m.close()


def test_native_input_exposes_unresolved_frontier_even_with_current_corridor(tmp_path):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        m.structure.data['unresolved_row_gaps'] = [unresolved_gap()]
        m._plan(p)
        assert any(c.get('observation_slot', '').startswith('scan_') for c in m.choices)
        folder = tmp_path/'frontier_input'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        assert prepared['payload']['state']['unresolved_row_gaps'][0]['id'] == 'gap_A_C'
        assert not prepared['payload']['state']['completion']['eligible']
        assert '未知区域，不是通道' in prepared['prompt']
    finally:
        m.close()


def test_prompt_projection_retains_all_candidates_and_fragmented_intervals():
    gaps = [[float(i)+.123456, float(i)+.423456] for i in range(10)]
    state = dict(candidates=[dict(candidate_id=f'choice_{i}', role='OBSERVE',
        target_world=[float(i)+.123456, 2., 3.], geometry=dict(distance=.123456789)) for i in range(30)],
        coverage=dict(corridors=[dict(corridor_id=f'C{i}', required=[[0., 12.]],
            uncovered=deepcopy(gaps), covered=[[10.5, 12.]]) for i in range(6)]),
        unresolved_row_gaps=[unresolved_gap()])
    before = deepcopy(state)
    projected = compact_prompt_state(state)
    assert state == before
    assert len(projected['candidate_rows']) == 30
    records = [dict(projected.get('candidate_defaults', {}), **dict(zip(projected['candidate_columns'], row)))
        for row in projected['candidate_rows']]
    assert [row['candidate_id'] for row in records] == [f'choice_{i}' for i in range(30)]
    assert all(row['geometry.distance'] == .123 for row in records)
    coverage = projected['coverage']
    corridors = [dict(zip(coverage['corridor_columns'], row)) for row in coverage['corridor_rows']]
    assert all(len(c['uncovered_mm']) == 10 and c['covered_mm'] == [[10500, 12000]] for c in corridors)
    assert corridors[0]['uncovered_mm'][0] == [123, 423]
    assert records[0]['target_world_mm'] == [123, 2000, 3000]
    assert projected['unresolved_row_gaps'][0]['id'] == 'gap_A_C'


def test_prompt_projection_shared_fields_and_return_runs_are_lossless():
    route = [['RETURN', None]]*62+[['REJOIN', 'C0']]*2
    state = dict(candidates=[dict(candidate_id=identifier, role='RETURN', allowed_behavior='RETURN',
        target_world=[1., 2., 3.], route_scope=deepcopy(route)) for identifier in ('path_A', 'path_B')])
    projected = compact_prompt_state(state)
    assert projected['candidate_defaults']['allowed_behavior'] == 'RETURN'
    records = [dict(projected['candidate_defaults'], **dict(zip(projected['candidate_columns'], row)))
        for row in projected['candidate_rows']]
    assert {c['candidate_id'] for c in records} == {'path_A', 'path_B'}
    for candidate in records:
        expanded = [[role, corridor] for role, corridor, count in candidate['route_scope_runs']
            for _ in range(count)]
        assert expanded == route
        assert candidate['target_world_mm'] == [1000, 2000, 3000]
    assert state['candidates'][0]['route_scope'] == route


def test_discovery_phase_cards_explain_every_option_without_ranking():
    candidates = [dict(candidate_id='away', relative_body_xyz_m=[0., -1.5, 0.],
        geometry=dict(median_structure_range_change_m=1., nearest_previous_observation_m=2.)),
        dict(candidate_id='toward', relative_body_xyz_m=[1.5, 0., 0.],
        geometry=dict(median_structure_range_change_m=-1., nearest_previous_observation_m=.01))]
    state = dict(intent='COVER', current_corridor=None, topology=[], candidates=candidates,
        observation_memory=dict(observed_structure=dict(count=3,
            centroid_body_xy_m=[8., 4.], centroid_bearing_body_deg=26.6)))
    phase, cards = discovery_phase_context(state)
    assert phase == 'FIND_CORRIDOR'
    assert cards.index('away：') < cards.index('toward：')
    assert '更远离观测结构' in cards and '更接近观测结构' in cards
    assert '距旧视点最近0.01米' in cards
    assert candidates[0]['candidate_id'] == 'away'
    assert 'recommended' not in state and 'chosen_id' not in state


def test_bootstrap_offers_full_body_circle_with_same_observation_checks(tmp_path):
    m = mission(tmp_path)
    try:
        m.active_corridor = None
        m.structure.data['corridors'] = []
        m._plan(np.array([0., 5., 2.5]))
        slots = {c['observation_slot'] for c in m.choices}
        assert slots == {'bootstrap_'+name for name in ('forward', 'forward_left', 'left',
            'back_left', 'back', 'back_right', 'right', 'forward_right')}
        assert all(c['requires_observed_connection'] and c['probe_key'] == 'bootstrap'
            for c in m.choices)
    finally:
        m.close()


def test_unauthorized_return_does_not_trigger_model_when_no_movement_remains(tmp_path):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        m.choices = [dict(m._anchor(m.home[:2], 'RETURN', None),
            candidate_id='RETURN_HOME', requested_intent='RETURN_HOME')]
        first = tmp_path/'unfinished'
        first.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', first, [], None)
        assert {c['candidate_id'] for c in prepared['candidates']} == {'RETURN_HOME', 'HOLD'}
        assert prepared['payload']['movement_alternatives'] == 0
        assert prepared['observation_only'] and not prepared['payload']['model_requested']
        m.intent = 'RETURN_HOME'
        second = tmp_path/'authorized'
        second.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_001', second, [], None)
        assert prepared['payload']['movement_alternatives'] == 1
        assert prepared['payload']['model_requested']
    finally:
        m.close()


def test_completed_end_exposes_each_candidates_coverage_and_route_scope(tmp_path):
    m = mission(tmp_path)
    p = np.array([20., 5., 4.5])
    try:
        m.ledger.covered['B'], m.ledger.directions['B'] = [[0., 20.]], [1., 0.]
        m._plan(p)
        folder = tmp_path/'completed_end'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        state = prepared['payload']['state']
        assert state['phase'] == 'CHANGE_OR_REPAIR' and state['at_confirmed_end'] == 'high'
        public = {c['candidate_id']:c for c in state['candidates']}
        # A finished aisle is not offered again while a connection exists.
        assert 'B_negative' not in public and 'B_positive' not in public
        assert public['C_via_high']['corridor_uncovered_m'] == 20.
        assert public['C_via_high']['route_scope'] == [['TRANSIT', 'B'], ['TRANSIT', 'C'], ['INSPECT', 'C']]
        assert 'C_via_high：behavior=CHANGE_AISLE' in prepared['payload']['prompt_phase_context']
        for candidate in public:
            assert candidate+'：' in prepared['payload']['prompt_phase_context']
        assert 'chosen_id' not in state and 'recommended' not in state
    finally:
        m.close()


def test_inspection_cards_distinguish_directional_gain_from_total_aisle_gap(tmp_path):
    m = mission(tmp_path)
    p = np.array([12., 5., 3.7])
    try:
        m.ledger.covered['B'], m.ledger.directions['B'] = [[0., 12.]], [1., 0.]
        m.last_completed_intention = dict(corridor_id='B', direction=1, actual_end=None)
        m._plan(p)
        folder = tmp_path/'directional_gain'; folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        state = prepared['payload']['state']
        public = {c['candidate_id']:c for c in state['candidates']}
        positive, negative = public['B_positive'], public['B_negative']
        assert positive['corridor_uncovered_m'] == negative['corridor_uncovered_m'] == 8.
        assert positive['new_coverage_m'] == 8. and positive['repeated_coverage_m'] == 0.
        assert negative['new_coverage_m'] == 0. and negative['repeated_coverage_m'] == 12.
        assert positive['total_route_distance_m'] > positive['explicit_route_distance_m']
        cards = prepared['payload']['prompt_phase_context']
        assert state['phase'] == 'INSPECT_CORRIDOR'
        assert 'B_positive：behavior=FOLLOW，可补巡8米' in cards
        assert 'B_negative：behavior=FOLLOW，可补巡0米' in cards
        assert '通道B，方向1，结束端None' in cards
        assert set(public) == {c['candidate_id'] for c in prepared['candidates']}
        compact = prepared['payload']['prompt_state']
        records = [dict(compact.get('candidate_defaults', {}), **dict(zip(
            compact['candidate_columns'], row))) for row in compact['candidate_rows']]
        projected = next(c for c in records if c['candidate_id'] == 'B_negative')
        assert projected['new_coverage_mm'] == 0 and projected['repeated_coverage_mm'] == 12000
        assert projected['inspection_range_s_mm'] == [12000, 0]
    finally:
        m.close()


def test_directional_gain_uses_fixed_ledger_frame_and_world_scope_after_refit():
    from omni_drones.navigation.forest.coverage import CoverageLedger
    angle = .04
    d, n = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
    c = corridor('B', 5.)
    c.update(direction=d.tolist(), normal=n.tolist())
    p = np.r_[10.*d+5.*n, 2.5]
    target = np.r_[14.*d+5.*n, 2.5]
    commitment = dict(corridor_id='B', direction=1, until_s=16.,
        scope=dict(kind='bounded_gap_repair', frame_direction=[1., 0.],
            endpoint_world_xy=[16., 5.], endpoint_s=16.))
    candidate = dict(target_world=target.tolist(), role='INSPECT', corridor_id='B',
        commitment=commitment)
    ledger = CoverageLedger(covered={'B':[[0., 12.]]}, directions={'B':[1., 0.]})
    metrics = candidate_intent_metrics(candidate, p, dict(corridors=[c]), ledger)
    assert metrics['inspection_range_s_m'] == [round(p[0], 3), 16.]
    assert metrics['new_coverage_m'] == round(4./d[0], 2)
    assert metrics['repeated_coverage_m'] == round((12.-p[0])/d[0], 2)
    assert metrics['total_route_distance_m'] == round((16.-p[0])/d[0], 2)
    assert commitment['scope']['endpoint_world_xy'] == [16., 5.]


def test_neighbor_gain_excludes_transit_and_boundary_probe_reports_missing_evidence():
    from omni_drones.navigation.forest.coverage import CoverageLedger
    observed = structure()
    p = np.array([20., 5., 2.5])
    choices = coverage_choices(observed, p, 'B', 0.,
        lambda xy, role, cid=None:dict(target_world=[*xy, 2.5], role=role, corridor_id=cid), {})
    selected = next(c for c in choices if c['candidate_id'] == 'C_via_high')
    ledger = CoverageLedger(covered={'C':[[12., 20.]]}, directions={'C':[1., 0.]})
    metrics = candidate_intent_metrics(selected, p, observed, ledger)
    assert metrics['inspection_range_s_m'] == [20., 0.]
    assert metrics['new_coverage_m'] == 12. and metrics['repeated_coverage_m'] == 8.
    assert metrics['explicit_route_distance_m'] == pytest.approx(9.4)
    assert metrics['total_route_distance_m'] == pytest.approx(29.4)
    observed['outer_sides']['high'] = dict(confirmed=False, observed_fraction=.72,
        terrain_supported_fraction=.86, diverse_views=True, positive_stem_counterexamples=[])
    outer = dict(target_world=[16., 15., 2.5], role='OBSERVE', commitment=dict(
        kind='outer_observation', side='high', corridor_id='C', direction=-1))
    metrics = candidate_intent_metrics(outer, p, observed, ledger)
    assert metrics['new_coverage_m'] is None and metrics['repeated_coverage_m'] is None
    assert metrics['boundary_observation'] == dict(side='high', confirmed=False,
        missing_ray_or_ground_fraction=.28, missing_ground_fraction=.14,
        diverse_views=True, positive_stem_counterexamples=0)


@pytest.mark.parametrize('sign', [-1, 1])
def test_bounded_gap_repair_keeps_world_endpoint_after_row_refit(tmp_path, sign):
    m = mission(tmp_path)
    try:
        m.ledger.covered['B'] = [[4., 20.]] if sign < 0 else [[0., 16.]]
        m.ledger.directions['B'] = [1., 0.]
        m._plan(np.array([10., 5., 3.5]))
        selected = next(c for c in m.choices if c['candidate_id'] == 'B_gap_repair_0')
        scope = selected['commitment']['scope']
        assert scope['kind'] == 'bounded_gap_repair'
        assert np.isclose(np.dot(scope['endpoint_world_xy'], scope['frame_direction']), scope['endpoint_s'])
        m.commitment = deepcopy(selected['commitment'])
        frozen = deepcopy(m.commitment)
        current = m.structure.data['corridors'][1]
        angle = .1
        current.update(direction=[np.cos(angle), np.sin(angle)], normal=[-np.sin(angle), np.cos(angle)],
            span=[-3., 25.])
        endpoint = np.asarray(scope['endpoint_world_xy'])
        p = np.r_[endpoint-np.array([sign, 0.]), 3.]
        m._plan(p)
        assert m.commitment == frozen
        target = np.asarray(m.route[0]['target_world'])
        assert np.isclose(target[:2]@scope['frame_direction'], scope['endpoint_s'])
        goal = m._native_goal(dict(candidate_id='continue_bound', behavior='FOLLOW'), m.route[0],
            'decision_000', 1., p, np.zeros(3), 0., snapshot(p), continuation=True)
        assert goal.completion_mode == 'stop' and m.continuation is None
    finally:
        m.close()


def test_gap_repair_public_input_retains_immutable_scope(tmp_path):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        m.ledger.covered['B'], m.ledger.directions['B'] = [[4., 20.]], [1., 0.]
        m._plan(p)
        selected = next(c for c in m.choices if c['candidate_id'] == 'B_gap_repair_0')
        folder = tmp_path/'repair_scope'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        public = next(c for c in prepared['payload']['state']['candidates'] if c['candidate_id'] == selected['candidate_id'])
        assert public['commitment_scope']['scope'] == selected['commitment']['scope']
        assert public['commitment_scope']['until_s'] == selected['commitment']['until_s']
        projected = prepared['payload']['prompt_state']
        records = [dict(zip(projected['candidate_columns'], row)) for row in projected['candidate_rows']]
        repair = next(c for c in records if c['candidate_id'] == selected['candidate_id'])
        scopes = [dict(zip(projected['scope_columns'], row)) for row in projected['scope_rows']]
        frozen = next(scope for scope in scopes if scope['scope_ref'] == repair['scope_ref'])
        assert frozen['scope.frame_direction'] == [1., 0.]
        assert frozen['scope.endpoint_world_xy'] == [0., 5.]
    finally:
        m.close()


def test_public_work_area_is_passed_unchanged_and_identifies_v2(tmp_path):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        public = dict(xy_bounds=[11., 108., -3., 40.], coverage_scope='BOUND_PUBLIC_WORK_AREA',
            source='operator supplied XY task rectangle')
        m.structure.data['public_work_area'] = deepcopy(public)
        m._plan(p)
        folder = tmp_path/'public_area'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        assert prepared['payload']['state']['public_work_area'] == public
        assert prepared['payload']['prompt_state']['public_work_area'] == public
        assert m.definition()['protocol'] == 'qwen_native_forest_s_coverage_v2'
    finally:
        m.close()


def test_staggered_ends_keep_the_change_aisle_route_through_its_headland(tmp_path):
    # Run 59fed373: the neighbour's end lies 6.1 m beyond the current one, so
    # the route's own headland stage stopped 8.3 m past the current end, the
    # next stage failed the 8 m check and no change was offered there again.
    m = mission(tmp_path)
    try:
        x, y = np.meshgrid(np.arange(-5., 37.), np.arange(-6., 17.))
        m.structure.data['terrain'] = dict(vertices=np.c_[x.ravel(), y.ravel(), .1*x.ravel()].tolist(),
            timestamps=[1.]*x.size)
        c = m.structure.data['corridors'][2]
        c['span'], c['ends']['high']['s'] = [0., 26.5], 26.5
        finish(m)
        p = np.array([20., 5., 4.5])
        m._plan(p)
        selected = next(c for c in m.choices if c['candidate_id'] == 'C_via_high')
        headland = np.asarray(selected['route'][0]['target_world'])
        assert headland[0]-20. > 8.
        assert m._activate(dict(candidate_id='C_via_high', behavior='CHANGE_AISLE'), selected,
            dict(task_revision=1, goal_revision=1, request_id='decision_000'),
            1., p, np.zeros(3), 0., snapshot(p)) is not None
        checks = {}
        assert m._validate(snapshot(headland), headland, 0., m.route[1], checks=checks)[0]
        assert checks['near_connection_end']
        m.route, m.commitment, m.choices = [], None, []
        m._plan(headland)
        assert 'C_via_high' in {c['candidate_id'] for c in m.choices}
    finally:
        m.close()


@pytest.mark.parametrize('x,near,far', [(0., 'low', 'high'), (20., 'high', 'low')])
def test_headland_connection_requires_nearby_end_at_offer_and_activation(tmp_path, x, near, far):
    m = mission(tmp_path)
    p = np.array([x, 5., 2.5+.1*x])
    try:
        finish(m)
        m.structure.data['rows'] = [dict(id='Ba', offset_m=2.5), dict(id='Bb', offset_m=7.5)]
        m._plan(p)
        ids = {c['candidate_id'] for c in m.choices}
        assert {'A_via_'+near, 'C_via_'+near, 'outer_low_via_'+near, 'outer_high_via_'+near} <= ids
        assert not any(name.endswith('_via_'+far) for name in ids)
        # A formerly valid asynchronous choice cannot remain valid after the
        # aircraft has moved to the other end of the same long corridor.
        stale = next(c for c in m.choices if c['candidate_id'] == 'C_via_'+near)
        moved = np.array([20.-x, 5., 2.5+.1*(20.-x)])
        checks = {}
        assert not m._validate(snapshot(moved), moved, 0., stale, checks=checks)[0]
        assert checks['near_connection_end'] is False
        assert m._activate(dict(candidate_id=stale['candidate_id'], behavior='CHANGE_AISLE'), stale,
            dict(task_revision=1, goal_revision=1, request_id='decision_000'),
            1., moved, np.zeros(3), 0., snapshot(moved)) is None
    finally:
        m.close()


def test_completed_intention_preserves_direction_source_and_actual_end(tmp_path):
    m = mission(tmp_path)
    p = np.array([0., 5., 2.5])
    try:
        m.commitment = dict(corridor_id='B', direction=-1,
            source_request_id='decision_007', candidate_id='B_negative')
        m.last_time = 42.
        m.ledger.covered['B'], m.ledger.directions['B'] = [[0., 20.]], [1., 0.]
        m._plan(p)
        previous = m.last_completed_intention
        assert m.commitment is None
        assert previous == dict(corridor_id='B', direction=-1,
            source_request_id='decision_007', candidate_id='B_negative', actual_end='low',
            end_confirmed=True, actual_position=p.tolist(), time=42., task_revision=1)
        folder = tmp_path/'last_direction'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_008', folder, [], None)
        state = prepared['payload']['state']
        assert state['previous_intention'] == previous and state['current_intention'] is None
        assert '通道B，方向-1，结束端low' in prepared['payload']['prompt_phase_context']
        selected = next(c for c in state['candidates'] if c['candidate_id'] == 'C_via_low')
        assert selected['allowed_behavior'] == 'CHANGE_AISLE'
        assert selected['connection_end'] == 'low'
        assert 0. < selected['reposition_distance_m'] < selected['total_route_distance_m']
        m.recovery_pending = True
        second = tmp_path/'recovery_direction'
        second.mkdir()
        recovered = m._prepare(snapshot(p), p, 0., 'decision_009', second, [], None)
        assert recovered['payload']['state']['phase'] == 'RECOVER'
        assert 'C_via_low：behavior=CHANGE_AISLE' in recovered['payload']['prompt_phase_context']
    finally:
        m.close()


@pytest.mark.parametrize('committed', [False, True])
@pytest.mark.parametrize('obstacle_distance', [.75, .825])
def test_occupied_inspection_endpoint_uses_shorter_clear_supported_anchor(tmp_path, committed, obstacle_distance):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        m.planning_clearance_reserve = .15
        # Physical clearance alone, or clearance without the planner's .05m
        # sampling pad, cannot authorize an unreachable stopping waypoint.
        observed = snapshot(p)
        observed['points'] = np.array([[14.+obstacle_distance, 5., 3.9]])
        m.latest = observed
        if committed:
            m.commitment = dict(corridor_id='B', direction=1,
                source_request_id='decision_000', candidate_id='B_positive')
        m._plan(p)
        selected = m.route[0] if committed else next(c for c in m.choices if c['candidate_id'] == 'B_positive')
        assert np.allclose(selected['target_world'], [13., 5., 3.8])
        # Reverse endpoints use the same world occupancy check, regardless of
        # the old BEV crop that starts only 2m behind the aircraft.
        assert m._endpoint_clear(m._endpoint_safety(observed, p, 0.), [6., 5., 3.1])
        assert np.array_equal(m._endpoint_safety(observed, p, 0.).frame_position, p)
    finally:
        m.close()


def test_inspection_endpoint_check_retains_a_star_detour_and_rechecks_new_hits(tmp_path):
    m = mission(tmp_path)
    p = np.array([10., 5., 3.5])
    try:
        observed = snapshot(p)
        observed['points'] = np.array([[12., 5., 3.7]])
        m.latest = observed
        m._plan(p)
        selected = next(c for c in m.choices if c['candidate_id'] == 'B_positive')
        assert np.allclose(selected['target_world'], [14., 5., 3.9])
        safety = m._endpoint_safety(observed, p, 0.)
        assert not safety.clear(m._lift_guidance(p, selected['target_world']))
        assert m._validate(observed, p, 0., selected)[0]
        fresh = dict(observed, points=np.array([selected['target_world']]))
        assert not m._validate(fresh, p, 0., selected)[0]
        assert m._native_goal(dict(candidate_id='B_positive', behavior='FOLLOW'), selected,
            'decision_000', 1., p, np.zeros(3), 0., fresh) is None
        assert m.rejections == 0 and m.status == 'WAITING_NEXT'
    finally:
        m.close()


def test_lateral_order_ignores_discovery_ids_and_flipped_normals():
    from omni_drones.navigation.forest.coverage_candidates import corridor_lateral_order
    # Discovery order R04 (outermost, found late) next to R01, as in run b5ce9972.
    corridors = [dict(corridor('R01_R02', 11.), id='R01_R02'), dict(corridor('R04_R01', 6.), id='R04_R01'),
                 dict(corridor('R03_R05', 21.), id='R03_R05', normal=[0., -1.], offset_m=-21.),
                 dict(corridor('R02_R03', 16.), id='R02_R03')]
    order = corridor_lateral_order(corridors)
    assert [order[i]['lateral_index'] for i in ('R04_R01', 'R01_R02', 'R02_R03', 'R03_R05')] == [1, 2, 3, 4]
    assert order['R04_R01']['observed_edge'] == 'low' and order['R03_R05']['observed_edge'] == 'high'
    assert order['R01_R02']['observed_edge'] is None
    assert corridor_lateral_order([corridor('A', 0.)])['A']['observed_edge'] == 'only'
    assert corridor_lateral_order([]) == {}


def test_model_input_exposes_lateral_order_and_observed_edges(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    try:
        m._plan(p)
        folder = tmp_path/'decision'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        payload = json.loads((folder/'input.json').read_text())
        topology = {t['corridor_id']:t for t in payload['state']['topology']}
        assert [topology[i]['lateral_index'] for i in 'ABC'] == [1, 2, 3]
        assert topology['A']['observed_edge'] == 'low' and topology['C']['observed_edge'] == 'high'
        assert payload['state']['current_corridor_lateral_index'] == 2
        assert payload['state']['observed_corridor_count'] == 3
        for c in payload['state']['candidates']:
            if c.get('corridor_id') in topology:
                assert c['corridor_lateral_index'] == topology[c['corridor_id']]['lateral_index']
        assert 'lateral_index' in prepared['prompt'] and '最外侧' in prepared['prompt']
        # Still no recommendation: the projection keeps every candidate, unranked.
        assert len(payload['prompt_state']['candidate_rows']) == len(payload['state']['candidates'])
    finally:
        m.close()


def test_unbound_entry_to_a_row_bordered_corridor_goes_around_its_confirmed_end():
    observed = structure()
    p = np.array([2., 2.5, 2.5])   # between A and B, bound to neither
    anchor = lambda xy, role, cid=None:dict(target_world=[*xy, 2.5], role=role, corridor_id=cid)
    choices = {c['candidate_id']:c for c in coverage_choices(observed, p, None, 0., anchor, {})}
    for identifier, offset in (('A', 0.), ('B', 5.)):
        entry = choices[identifier+'_entry_via_low']
        route = np.array([point['target_world'] for point in entry['route']])
        np.testing.assert_allclose(route[:, :2], [[-2.2, 2.5], [-2.2, offset]])
        assert [point['role'] for point in entry['route']] == ['OBSERVE', 'OBSERVE']
        assert entry['route'][-1]['corridor_id'] == identifier
        assert entry['requires_observed_connection'] and entry['confirmed_ends'] == [[identifier, 'low']]
    # Far from the high ends (s=20 is 18 m away) and C is > 8 m from neither end... only nearby ends.
    assert not any(key.endswith('_entry_via_high') for key in choices)
    # The straight short entries remain separate alternatives.
    assert any(key.startswith('A_entry_observe_') for key in choices)


def test_unconfirmed_or_aligned_corridor_gets_no_headland_entry():
    observed = structure(confirmed=False)
    anchor = lambda xy, role, cid=None:dict(target_world=[*xy, 2.5], role=role, corridor_id=cid)
    choices = coverage_choices(observed, np.array([2., 2.5, 2.5]), None, 0., anchor, {})
    assert not any('_entry_via_' in c['candidate_id'] for c in choices)
    choices = coverage_choices(structure(), np.array([2., 5.3, 2.5]), None, 0., anchor, {})
    assert not any(c['candidate_id'].startswith('B_entry_via_') for c in choices)


def test_recovery_budget_is_restored_only_after_consecutive_completed_goals(tmp_path):
    # Run e1f4e618 spent all three recoveries in the first 154 s, so a block at
    # 2879 s ended a mission that had kept completing goals in between.
    m = mission(tmp_path)
    m.recovery_refill_goals = 2
    p = np.array([10., 5., 3.5])
    def complete(t):
        m._results.append(dict(goal_id=f'done_{t}', status='COMPLETE', reason='goal_reached'))
        m.update(t, p, np.zeros(3), 0.)
    try:
        for index in range(2):
            activate_blocked_monitor(m, float(index))
            m.update(float(index), p, np.zeros(3), 0.)
            m.update(index+.3, p, np.zeros(3), 0.)
        assert m.recoveries == 2 and m.status != 'FAILED'
        complete(5.)
        assert m.recoveries == 2   # one completed goal is not enough
        activate_blocked_monitor(m, 6.)
        m.update(6., p, np.zeros(3), 0.)
        m.update(6.3, p, np.zeros(3), 0.)
        assert m.recoveries == 3   # the new recovery resets the progress count
        complete(7.)
        assert m.recoveries == 3
        complete(8.)
        assert m.recoveries == 0
        assert [e['kind'] for e in m.events].count('coverage_recovery_budget_restored') == 1
    finally:
        m.close()


def test_guidance_height_changes_gently_from_the_actual_height(tmp_path):
    # Run 401c7d8c: a 0.75 m blend climbed steeply right in front of the
    # vehicle, into space a roll-180 MID-360 never observes.
    m = mission(tmp_path)
    try:
        m.max_climb_slope = .15
        start, target = np.array([2., 5., 3.0]), np.array([10., 5., 3.5])
        guide = m._lift_guidance(start, target)
        np.testing.assert_allclose(guide[0], start)
        np.testing.assert_allclose(guide[-1], target)
        run = np.linalg.norm(np.diff(guide[:, :2], axis=0), axis=1)
        assert (np.abs(np.diff(guide[:, 2])) <= .15*run+1e-9).all()
        # Terrain is z=0.1x: far from both ends it still holds ground + 2.5 m.
        middle = guide[len(guide)//2]
        assert abs(middle[2]-(.1*middle[0]+2.5)) < .35
        m.max_climb_slope = None
        steep = m._lift_guidance(start, target)
        assert np.abs(np.diff(steep[:, 2])).max() > .15*.5
    finally:
        m.close()


def test_missing_ground_holds_the_current_height_like_catkin_ws(tmp_path):
    # Run 0c23d8ef: a level upright MID-360 at 2 m AGL saw no ground within
    # ~17 m, so requiring observed ground rejected every move at the start.
    m = mission(tmp_path)
    try:
        m.structure.data['terrain'] = dict(vertices=[], timestamps=[])
        m.latest_position = np.array([2., 5., 3.2])
        with pytest.raises(ValueError, match='NO_OBSERVED_GROUND'):
            m._anchor([6., 5.], 'INSPECT', 'B')
        m.ground_fallback = 'hold_current'
        assert m._anchor([6., 5.], 'INSPECT', 'B')['target_world'][2] == pytest.approx(3.2)
        guide = m._lift_guidance(np.array([2., 5., 3.2]), np.array([6., 5., 3.2]))
        np.testing.assert_allclose(guide[:, 2], 3.2)
        assert m.ground_fallback_points > 0 and m.definition()['ground_fallback'] == 'hold_current'
    finally:
        m.close()


def test_known_terrain_sets_heights_without_any_observed_ground(tmp_path):
    # Sim stand-in for an airframe AGL measurement (deployment M300 RTK).
    m = mission(tmp_path)
    try:
        m.structure.data['terrain'] = dict(vertices=[], timestamps=[])
        xs, ys = np.arange(-10., 31.), np.arange(-10., 21.)
        m.terrain_truth = dict(xs=xs, ys=ys, heights=.2*xs[:, None]+0.*ys[None, :])
        assert m._anchor([6., 5.], 'INSPECT', 'B')['target_world'][2] == pytest.approx(.2*6.+2.5)
        guide = m._lift_guidance(np.array([2., 5., 2.9]), np.array([6., 5., 3.7]))
        assert m.ground_fallback_points == 0 and m.definition()['terrain_source'] == 'truth'
        np.testing.assert_allclose(guide[[0, -1], 2], [2.9, 3.7])
    finally:
        m.close()


def test_outer_flank_stays_inside_public_work_area(tmp_path):
    m = mission(tmp_path)
    p = np.array([20., 5., 4.5])
    try:
        finish(m)
        m.structure.data['rows'] = [dict(id='Ba', offset_m=2.5), dict(id='Bb', offset_m=7.5)]
        m.structure.data['public_work_area'] = dict(xy_bounds=[-5., 25., 1.5, 9.],
            coverage_scope='BOUND_PUBLIC_WORK_AREA', source='test')
        m._plan(p)
        outer = [c for c in m.choices if c['candidate_id'].startswith('outer_')]
        # Both flanks are under 1.5 m wide inside the job: observed from the aisle.
        assert not outer
        m.structure.data['public_work_area']['xy_bounds'] = [-5., 25., -3., 9.]
        m._plan(p)
        low = next(c for c in m.choices if c['candidate_id'] == 'outer_low_via_high')
        assert all(point['target_world'][1] >= -2.5 for point in low['route'])
    finally:
        m.close()


def test_covered_aisle_with_open_end_asks_to_confirm_it():
    from omni_drones.navigation.forest.coverage_candidates import discovery_phase_context
    state = dict(intent='COVER', current_corridor='B', current_corridor_uncovered_m=0.,
        at_confirmed_end=None, unconfirmed_near_end='high', candidates=[])
    phase, text = discovery_phase_context(state)
    assert phase == 'CONFIRM_END' and 'observe_high' in text and '尚未巡完' not in text


def test_s_order_prefers_uncovered_neighbor_and_heads_toward_open_aisle(tmp_path):
    m = mission(tmp_path)
    p = np.array([15., 5., 4.])
    try:
        finish(m, 'B'); finish(m, 'A')
        m._plan(p)
        ids = {c['candidate_id'] for c in m.choices}
        # A is finished, C is not: only the useful aisle change remains.
        assert 'C_via_high' in ids and 'A_via_high' not in ids
        # Only finished neighbours, D open beyond C: transfer straight to D.
        finish(m, 'C')
        m.structure.data['corridors'].append(corridor('D', 15.))
        m._plan(p)
        ids = {c['candidate_id'] for c in m.choices}
        assert 'D_via_high' in ids and 'C_via_high' not in ids and 'A_via_high' not in ids
    finally:
        m.close()


def test_coverage_stall_fails_fast_without_new_coverage(tmp_path):
    from omegaconf import OmegaConf
    from pathlib import Path
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2]/'cfg/task/ForestSurvey.yaml')
    m = CoverageMission(RuleWorker(), '沿S形巡检整个树林，然后返回起点并停稳。',
        [0., 5., 2.5], 0., SimpleNamespace(**dict(cfg.flight)), tmp_path,
        [-20., 40., -20., 30.], [-10., 30.], Structure(structure()),
        decision_policy='qwen', desired_agl=2.5, max_decisions=100, coverage_stall_seconds=10.)
    try:
        m.status, m.intent = 'RUNNING', 'COVER'
        p = np.array([-5., 5., 2.5])
        for now in (1., 5., 10.):
            m.update(now, p, np.zeros(3), 0.)
            assert m.status != 'FAILED'
        m.update(12., p, np.zeros(3), 0.)
        assert m.status == 'FAILED' and m.reason == 'coverage_stalled'
    finally:
        m.close()


def test_completed_coverage_offers_only_returns_including_headland(tmp_path):
    m = mission(tmp_path)
    p = np.array([20., 10., 4.5])
    try:
        for corridor_id in 'ABC':
            finish(m, corridor_id)
        m.structure.data['outer_sides'] = dict(low=dict(confirmed=True), high=dict(confirmed=True))
        m.active_corridor = 'C'
        m.breadcrumbs += [[10., 5., 3.5], [20., 5., 4.5], [20., 10., 4.5]]
        m.latest_position = p
        m._plan(p)
        ids = {c['candidate_id'] for c in m.choices}
        assert ids == {'RETURN_HOME_HEADLAND', 'RETURN_HOME'}
        headland = next(c for c in m.choices if c['candidate_id'] == 'RETURN_HOME_HEADLAND')
        xy = np.asarray([point['target_world'][:2] for point in headland['route']])
        assert headland['headland_side'] == 'high'
        assert np.all(np.linalg.norm(np.diff(np.vstack([p[:2], xy]), axis=0), axis=1) <= 4.+1e-9)
        # Outside every row end (high ends at s=20) until the lateral move is done.
        assert xy[:-1, 0].max() >= 22.2-1e-9 and np.allclose(headland['route'][-1]['target_world'], m.home)
        folder = tmp_path/'return'
        folder.mkdir()
        prepared = m._prepare(snapshot(p), p, 0., 'decision_000', folder, [], None)
        assert prepared['payload']['state']['phase'] == 'RETURN'
        assert 'RETURN_HOME_HEADLAND：behavior=RETURN' in prepared['payload']['prompt_phase_context']
    finally:
        m.close()


def test_structure_flicker_at_a_through_gate_brakes_instead_of_failing(tmp_path):
    m = mission(tmp_path)
    p = np.array([18., 5., 4.3])
    try:
        activate_blocked_monitor(m, 0., through=True)
        m.status, m.route = 'WAITING_NEXT', []
        m.structure.data['identity_ambiguous'] = True
        assert m.advance(1., p, np.array([.8, 0., 0.]), 0.) is None
        assert m.status != 'FAILED' and m.requires_handoff_stop and m.commitment is None
        assert [e['kind'] for e in m.events[-2:]] == ['commitment_dropped', 'continuation_suspended']
        assert m.advance(1.1, p, np.array([.5, 0., 0.]), 0.) is None and m.decisions == 0
        # Settled and the structure is consistent again: an ordinary new decision.
        m.structure.data['identity_ambiguous'] = False
        m.latest = snapshot(p)
        m.latest['timestamp'] = 2.
        assert m.advance(2., p, np.zeros(3), 0., reference_settled=True) is None
        assert m.decisions == 1 and not any(e['kind'] == 'task_failed' for e in m.events)
    finally:
        m.close()


@pytest.mark.parametrize('covered,confirmed,kept', [([[0., 1.]], False, False), ([[0., 1.]], True, True),
                                                    ([[0., 5.], [12., 20.]], False, True)])
def test_gap_repair_to_an_unconfirmed_span_end_is_left_to_open_ended_inspection(
        tmp_path, covered, confirmed, kept):
    m = mission(tmp_path)
    p = np.array([1., 5., 2.6])
    try:
        m.structure.data['corridors'][1]['ends']['high']['confirmed'] = confirmed
        m.ledger.covered['B'], m.ledger.directions['B'] = covered, [1., 0.]
        m._plan(p)
        ids = {c['candidate_id'] for c in m.choices}
        assert 'B_positive' in ids
        # Up to a still-growing end it would duplicate B_positive, then stop mid-aisle.
        assert ('B_gap_repair_0' in ids) == kept
    finally:
        m.close()


def test_motion_while_stopping_after_an_inspection_goal_is_credited(tmp_path):
    # Run c1c3147f: 1.0 m flown while the vehicle stopped between two
    # continuations was never credited, and Qwen re-flew the aisle for it.
    m = mission(tmp_path)
    try:
        m._selected = dict(candidate_id='B_positive', behavior='FOLLOW', role='INSPECT', corridor_id='B')
        m.requires_handoff_stop = True
        for k, x in enumerate(np.linspace(6., 7., 11)):
            m.latest['timestamp'], m.latest['captured'] = .1*k, time.perf_counter()
            m.update(.1*k, np.array([x, 5., 3.5]), np.array([1., 0., 0.]), 0.)
        assert m.status == 'WAITING_NEXT'
        np.testing.assert_allclose(m.ledger.covered['B'], [[6., 7.]])
        # Measured motion only: a stale observation credits nothing.
        m.latest['timestamp'] = -5.
        m.update(1.1, np.array([7.1, 5., 3.5]), np.array([1., 0., 0.]), 0.)
        m.update(1.2, np.array([7.2, 5., 3.5]), np.array([1., 0., 0.]), 0.)
        np.testing.assert_allclose(m.ledger.covered['B'], [[6., 7.]])
    finally:
        m.close()


def test_a_stop_without_a_fresh_observation_waits_instead_of_replanning_each_tick(tmp_path, monkeypatch):
    # Run 8ba92099: a recovery stop replanned on every 10 ms tick with a stale
    # observation; the worker slowed until every result was stale, 150 s.
    m = mission(tmp_path)
    p = np.array([18., 5., 4.3])
    calls = []
    try:
        activate_blocked_monitor(m, 0.)
        m.status, m.route, m.requires_handoff_stop = 'WAITING_NEXT', [], True
        monkeypatch.setattr(m, '_plan', lambda position: calls.append(position))
        m.latest['timestamp'] = -1.
        assert m.advance(1., p, np.zeros(3), 0.) is None and calls == []
        m.latest['timestamp'], m.latest['captured'] = 1., time.perf_counter()
        m.advance(1., p, np.zeros(3), 0.)
        assert len(calls) == 1
    finally:
        m.close()
