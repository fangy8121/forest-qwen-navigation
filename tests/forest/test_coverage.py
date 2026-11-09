import json
from types import SimpleNamespace
import numpy as np
import pytest
from omni_drones.navigation.forest.coverage import CoverageLedger, subtract, completion_evidence
from omni_drones.navigation.forest.qwen_subgoal import parse_coverage_command
from omni_drones.navigation.forest.coverage_mission import CoverageMission
from omni_drones.navigation.forest.coverage_flight import RuleWorker


def corridor(name='R01_R02', offset=0.):
    return dict(id=name, direction=[1., 0.], normal=[0., 1.], offset_m=offset,
        span=[0., 10.], ends={k:dict(confirmed=True, s=s) for k, s in [('low', 0.), ('high', 10.)]})


def test_interruption_leaves_gap_and_resumption_can_fill_it():
    ledger = CoverageLedger()
    c = corridor()
    for i, x in enumerate(np.arange(0., 4.01, .1)):
        ledger.observe(i*.1, [x, 0., 2.], [c])
    for i, x in enumerate(np.arange(4., 7.01, .1)):
        ledger.observe(5+i*.1, [x, 5., 2.], [c], enabled=False)
    for i, x in enumerate(np.arange(7., 10.01, .1)):
        ledger.observe(10+i*.1, [x, 0., 2.], [c])
    assert np.allclose(ledger.gaps(c), [[4., 7.]])
    for i, x in enumerate(np.arange(7., 3.89, -.1)):
        ledger.observe(20+i*.1, [x, 0., 2.], [c])
    assert ledger.gaps(c) == []


def test_adjacent_aisle_and_teleport_never_fill_original_gap():
    ledger = CoverageLedger()
    a, b = corridor(), corridor('R02_R03', 5.)
    ledger.observe(0., [0., 0., 2.], [a, b])
    ledger.observe(.1, [10., 0., 2.], [a, b])
    for i, x in enumerate(np.arange(0., 10.01, .1)):
        ledger.observe(1+i*.1, [x, 5., 2.], [a, b])
    assert ledger.gaps(a) == [[0., 10.]]
    assert ledger.gaps(b) == []


def test_row_refit_does_not_credit_stationary_world_position_as_motion():
    ledger = CoverageLedger()
    a = corridor(offset=50.)
    ledger.observe(0., [0., 50., 2.], [a])
    b = corridor(offset=50.*np.cos(.1))
    b.update(direction=[np.cos(.1), np.sin(.1)], normal=[-np.sin(.1), np.cos(.1)], span=[0., 15.])
    ledger.observe(.1, [0., 50., 2.], [b])
    assert not ledger.covered.get(a['id'])


def test_structure_accumulation_runs_in_separate_process(tmp_path):
    import os
    from omni_drones.navigation.forest.online_structure import OnlineStructure
    structure = OnlineStructure(tmp_path)
    frame = dict(timestamp=np.array(0.), position=np.array([0., 0., 2.]),
        points_sensor=np.empty((0, 3)), valid=np.zeros(0, bool), distances=np.empty(0),
        directions_world=np.empty((0, 3)), sensor_position=np.array([0., 0., 2.]),
        sensor_quaternion=np.array([1., 0., 0., 0.]), max_range=np.array(15.))
    try:
        structure.submit(frame)
        result = structure.future.result(timeout=30.)
        assert result['worker_pid'] != os.getpid()
        structure.submit(frame)
        assert structure.snapshot()['worker_pid'] == result['worker_pid']
    finally:
        structure.close()


def test_newly_discovered_span_reopens_coverage():
    c = corridor()
    ledger = CoverageLedger(covered={c['id']:[[0., 10.]]})
    assert ledger.gaps(c) == []
    c['span'] = [-3., 12.]
    assert ledger.gaps(c) == [[-3., 0.], [10., 12.]]


def test_small_interior_sensor_gaps_are_not_erased_by_arrival_tolerance():
    c = corridor()
    ledger = CoverageLedger(covered={c['id']:[[.1, 4.], [4.2, 9.9]]})
    assert np.allclose(ledger.gaps(c), [[4., 4.2]])


def test_completion_requires_outer_boundary_and_identity_evidence():
    c = corridor()
    ledger = CoverageLedger(covered={c['id']:[[0., 10.]]})
    s = dict(corridors=[c], outer_sides={})
    assert not completion_evidence(s, ledger)['eligible']
    s['outer_sides'] = dict(low=dict(confirmed=True), high=dict(confirmed=True))
    assert completion_evidence(s, ledger)['eligible']
    s['identity_ambiguous'] = True
    assert not completion_evidence(s, ledger)['eligible']


def test_new_evidence_revokes_completion_and_automatic_return(tmp_path):
    m = mission(tmp_path)
    try:
        m.coverage_result = 'COMPLETE_UNDER_ASSUMPTIONS'
        m.intent, m.automatic_return = 'RETURN_HOME', True
        m.update(1., np.array([0., 0., 2.]), np.zeros(3), 0.)
        assert m.coverage_result == 'IN_PROGRESS'
        assert m.intent == 'COVER' and m.status == 'WAITING_NEXT'
        assert m.task_revision == 2
    finally:
        m.close()


def test_coverage_reopen_cannot_clear_latched_structural_failure(tmp_path):
    m = mission(tmp_path)
    try:
        m.coverage_result, m.intent, m.automatic_return = 'COMPLETE_UNDER_ASSUMPTIONS', 'RETURN_HOME', True
        m.structure.last_error = 'structure budget'
        m.update(1., np.array([0., 0., 2.]), np.zeros(3), 0.)
        assert m.status == 'FAILED'
        assert m.reason.startswith('STRUCTURE:')
    finally:
        m.close()


@pytest.mark.parametrize('value', [
    dict(intent='REVISIT_CORRIDOR', corridor_id=None, resume_after=True),
    dict(intent='RETURN_HOME', corridor_id='R01_R02', resume_after=False),
    dict(intent='COVER', corridor_id=None, resume_after='false'),
    dict(intent='COVER', corridor_id=None, resume_after=True),
])
def test_invalid_command_cannot_become_execution_intent(value):
    with pytest.raises(ValueError):
        parse_coverage_command(json.dumps(value))


class Structure:
    last_error = None
    def snapshot(self):
        return dict(corridors=[corridor()], rows=[], outer_sides={})


def mission(tmp_path):
    return CoverageMission(RuleWorker(), '巡检整个树林', [0., 0., 2.], 0., SimpleNamespace(),
        tmp_path, [-15., 25., -15., 25.], [1., 3.], Structure())


def test_moving_pause_invalidates_pending_decision_and_preserves_coverage(tmp_path):
    m = mission(tmp_path)
    try:
        m.latest_speed = .4
        m.pending = dict(task_revision=1)
        m.ledger.covered['R01_R02'] = [[0., 4.]]
        assert m.receive('pause1', '暂停', 12.)
        assert m.status == 'WAITING_NEXT' and m.task_revision == 2
        assert m.intent == 'PAUSE' and not m.allows_stop()
        assert m.ledger.gaps(corridor()) == [[4., 10.]]
        assert not m.receive('pause1', '暂停', 12.1)
        assert m._results[-1]['status'] == 'CANCELLED'
        assert m.events[-1]['actual_speed_mps'] == .4
    finally:
        m.close()


def test_scheduled_resume_waits_for_actual_pause_reception(tmp_path):
    m = mission(tmp_path)
    try:
        m.command_schedule = [dict(command_id='p', text='暂停', at=1., min_speed=.1),
            dict(command_id='r', text='继续', at=0., after_command_id='p', delay=5.)]
        m.update(1., np.array([0., 0., 2.]), np.zeros(3), 0.)
        assert not m.seen_commands
        m.update(2., np.array([.4, 0., 2.]), np.array([.4, 0., 0.]), 0.)
        assert m.seen_commands == {'p'}
        m.update(6., np.array([.4, 0., 2.]), np.zeros(3), 0.)
        assert m.seen_commands == {'p'}
        m.update(7.1, np.array([.4, 0., 2.]), np.zeros(3), 0.)
        assert m.seen_commands == {'p', 'r'}
    finally:
        m.close()


def test_partial_command_append_waits_for_newline(tmp_path):
    m = mission(tmp_path)
    try:
        m.command_file.write_text('{"command_id":"x","text":"暂停"}')
        m._read_commands(1.)
        assert m.task_revision == 1
        with m.command_file.open('a') as f:
            f.write('\n')
        m._read_commands(2.)
        assert m.task_revision == 2
    finally:
        m.close()


def test_pause_resume_restores_return_intent_instead_of_starting_new_survey(tmp_path):
    m = mission(tmp_path)
    try:
        m.intent = 'RETURN_HOME'
        m.receive('p', '暂停', 1.)
        m.receive('r', '继续', 2.)
        assert not m._command_tick(2.)
        assert m._command_tick(2.1)
        assert m.intent == 'RETURN_HOME'
    finally:
        m.close()


def test_pause_resume_preserves_revisit_target_and_remaining_semantics(tmp_path):
    m = mission(tmp_path)
    try:
        m.intent = 'REVISIT_CORRIDOR'
        m.preferred_corridor, m.active_corridor = 'R01_R02', 'R02_R03'
        m.resume_after = True
        m.resume_context = dict(intent='COVER', corridor_id='R02_R03')
        m.receive('p', '暂停', 1.)
        m.receive('r', '继续', 2.)
        m._command_tick(2.)
        assert m._command_tick(2.1)
        assert m.intent == 'REVISIT_CORRIDOR'
        assert m.preferred_corridor == 'R01_R02' and m.resume_after
        assert m.resume_context['corridor_id'] == 'R02_R03'
        assert m.active_corridor == 'R02_R03'
    finally:
        m.close()


def test_previous_corridor_reference_is_frozen_at_command_reception(tmp_path):
    m = mission(tmp_path)
    try:
        m.active_corridor, m.departed_corridor = 'R02_R03', 'R01_R02'
        m.entered_corridors = ['R01_R02', 'R02_R03']
        m.receive('visit', '回刚才那条通道，然后继续', 1.)
        m.active_corridor, m.departed_corridor = 'R03_R04', 'R02_R03'
        m._command_tick(2.)
        assert m._command_tick(2.1)
        assert m.preferred_corridor == 'R01_R02'
    finally:
        m.close()


def test_return_uses_observed_loop_closure_instead_of_replaying_entire_survey(tmp_path):
    m = mission(tmp_path)
    try:
        m.breadcrumbs = [[0., 0., 2.], [3., 0., 2.], [3., 3., 2.], [0., 3., 2.], [3., 0., 2.], [6., 0., 2.]]
        m.latest_position = np.array([6., 0., 2.])
        route = m._history_route()
        assert [c['target_world'] for c in route] == [[3., 0., 2.], [0., 0., 2.]]
    finally:
        m.close()


def test_history_route_coalesces_straight_stops_but_keeps_turns(tmp_path):
    m = mission(tmp_path)
    try:
        m.breadcrumbs = [[float(x), 0., 2.] for x in range(7)]
        m.latest_position = np.array([6., 0., 2.])
        route = m._history_route()
        assert len(route) == 2 and route[-1]['target_world'] == [0., 0., 2.]
        m.breadcrumbs = [[0., 0., 2.], [1., 0., 2.], [2., 0., 2.], [2., 1., 2.], [2., 2., 2.]]
        m.latest_position = np.array([2., 2., 2.])
        route = m._history_route()
        assert route[0]['target_world'] == [2., 0., 2.]
    finally:
        m.close()


def test_failed_final_model_request_cannot_be_reported_as_passed():
    from omni_drones.navigation.forest.coverage_flight import evaluate_coverage
    x = np.linspace(0., 10., 101)
    actual = np.c_[x, np.zeros(len(x)), np.full(len(x), 2.)]
    m = SimpleNamespace(coverage_result='COMPLETE_UNDER_ASSUMPTIONS', status='FAILED', stop_requested=False)
    result = evaluate_coverage(actual, np.zeros(3), actual[-1], True, m, False,
        dict(corridors=[dict(id='truth', y=0., x_range=[0., 10.])]))
    assert result['actual_full_route_coverage']
    assert result['final_home'] and result['final_measured_stop']
    assert not result['full_coverage_passed']


def test_temporary_corridor_must_be_filled_before_original_task_is_restored(tmp_path):
    m = mission(tmp_path)
    try:
        m.temporary_corridor = 'R01_R02'
        m.resume_context = dict(intent='COVER', corridor_id='R02_R03')
        m.ledger.covered['R01_R02'] = [[0., 4.], [7., 10.]]
        m._plan(np.array([7., 0., 2.]))
        assert m.temporary_corridor == 'R01_R02' and m.route
        m.route = []
        m.ledger.covered['R01_R02'] = [[0., 10.]]
        m._plan(np.array([4., 0., 2.]))
        assert m.temporary_corridor is None
        assert m.preferred_corridor == 'R02_R03'
        assert m.events[-1]['kind'] == 'interrupted_task_restored'
    finally:
        m.close()


def test_final_coverage_return_does_not_inherit_temporary_resume_flag(tmp_path):
    m = mission(tmp_path)
    try:
        c = corridor()
        state = dict(corridors=[c], rows=[], outer_sides={k:dict(confirmed=True) for k in ('low', 'high')})
        m.structure = SimpleNamespace(snapshot=lambda:state, last_error=None)
        m.ledger.covered[c['id']] = [[0., 10.]]
        m.resume_after = True
        m._plan(np.array([0., 0., 2.]))
        assert m.intent == 'RETURN_HOME' and not m.resume_after
        m._plan(np.array([0., 0., 2.]))
        assert m.intent == 'FINISH'
    finally:
        m.close()


def test_online_candidate_renders_with_existing_bev_protocol(tmp_path):
    from omegaconf import OmegaConf
    from pathlib import Path
    m = mission(tmp_path)
    try:
        cfg = OmegaConf.load(Path(__file__).resolve().parents[2]/'cfg/task/ForestSurvey.yaml')
        m.flight = SimpleNamespace(**dict(cfg.flight))
        m.route = [m._anchor([2., 0.], 'INSPECT', 'R01_R02')]
        folder = tmp_path/'test_input'
        folder.mkdir()
        snapshot = dict(points=np.array([[10., 10., 0.]]), evidence=np.array([[0., 0., 2.]]), timestamp=1.)
        result = m._prepare(snapshot, np.array([0., 0., 2.]), 0., 'test_input', folder, [], None)
        assert Path(result['image']).is_file()
        assert [c['kind'] for c in result['candidates']] == ['observed_task_anchor', 'hold']
    finally:
        m.close()
