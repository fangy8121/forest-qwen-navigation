"""Cross-boundary task tests: sensed state, asynchronous policy and evaluation.

The fake model only chooses IDs; it cannot directly move the measured vehicle,
complete a goal, or supply the independent task evaluator's required direction.
"""
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from omni_drones.navigation.forest.task_supervisor import QwenTaskMission, evaluate_direction_task
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


HOME = np.array([0., 0., 2.])
ZERO = np.zeros(3)


class FakeWorker:
    def __init__(self):
        self.requests, self.responses = [], []

    def submit(self, image, prompt, *, request_id):
        self.requests.append(dict(image=image, prompt=prompt, request_id=request_id))
        return True

    def poll(self):
        return self.responses.pop(0) if self.responses else None

    def answer(self, candidate_id, behavior='FOLLOW', *, request_id=None):
        self.responses.append(dict(request_id=request_id or self.requests[-1]['request_id'],
            raw=json.dumps(dict(candidate_id=candidate_id, behavior=behavior)),
            decision=dict(candidate_id=candidate_id, behavior=behavior),
            latency_seconds=.01, input_tokens=100, output_tokens=15))


def snapshot(now, *, captured=None):
    return dict(timestamp=now, captured=time.perf_counter() if captured is None else captured,
                points=np.empty((0, 3)), evidence=np.empty((0, 3)))


def pilot_candidates(*args, **kwargs):
    return [dict(candidate_id=identifier, target_world=target) for identifier, target in (
        ('LEFT', [0., 2., 2.]), ('RIGHT', [0., -2., 2.]),
        ('HOME', HOME.tolist()), ('HOLD', HOME.tolist()))]


@pytest.fixture
def pilot(tmp_path):
    worker = FakeWorker()
    mission = None

    def prepare(obs, position, yaw, request_id, folder, history, rejection):
        # The production render/model path is tested separately. This seam
        # preserves causal candidates and actual history for state-machine QA.
        image = folder/'input.png'
        image.write_bytes(b'fake rendered image')
        candidates = mission._candidates(obs, position, yaw)
        return dict(image=str(image), prompt=json.dumps(dict(history=history, candidates=candidates)),
                    candidates=candidates)

    mission = QwenTaskMission(worker, '先向起飞朝向左侧飞行两米，返回起点并停稳。',
        HOME, 0., DEFAULT_FLIGHT, tmp_path, (-5., 5., -5., 5.), (1., 3.),
        prepare_fn=prepare, candidate_fn=pilot_candidates)
    mission._validate = lambda obs, position, yaw, selected: (True, obs, position)
    yield mission, worker
    mission.close()


def warmup(mission):
    mission.update(0., HOME, ZERO, 0.)
    mission.update(.6, HOME, ZERO, 0.)
    assert mission.status == 'WAITING_NEXT'


def tick(mission, position=HOME, *, fresh=True, reference_settled=True):
    now = mission.last_time+.005
    if fresh:
        mission.observe(snapshot(now))
    return mission.advance(now, position, ZERO, 0., reference_settled=reference_settled)


def until(mission, condition, position=HOME, timeout=2.):
    deadline = time.perf_counter()+timeout
    while time.perf_counter() < deadline:
        tick(mission, position)
        if condition():
            return
        time.sleep(.001)
    raise AssertionError(f'condition timed out: {mission.feedback()}')


def requested(mission, worker, count=1, position=HOME):
    until(mission, lambda: len(worker.requests) == count, position)


def test_input_failure_budget_does_not_queue_another_request_after_failure(pilot):
    mission, worker = pilot
    warmup(mission)
    mission.max_rejections = 1
    def broken(*args):
        raise OSError('input image unavailable')
    mission._prepare_fn = broken
    until(mission, lambda: mission.status == 'FAILED')
    assert mission.reason == 'policy_rejection_budget'
    assert mission.preparation is None and mission.pending is None
    assert worker.requests == []
    assert len([e for e in mission.events if e['kind'] == 'policy_requested']) == 1


def activate(mission, worker, candidate='LEFT', behavior='FOLLOW', position=HOME):
    requested(mission, worker, len(worker.requests)+1, position)
    worker.answer(candidate, behavior)
    until(mission, lambda: mission.status == 'RUNNING', position)
    return mission.current_goal


def measured_arrival(mission, position):
    now = mission.last_time+.1
    mission.update(now, position, ZERO, 0.)
    assert mission.status == 'RUNNING'
    mission.update(now+.6, position, ZERO, 0.)
    assert mission.status == 'WAITING_NEXT'


def test_warmup_is_excluded_and_physical_reference_must_settle(pilot):
    mission, worker = pilot
    warmup(mission)
    assert mission.history == []
    assert mission.feedback()['completed_movement_goals'] == 0
    for _ in range(10):
        assert tick(mission, reference_settled=False) is None
    assert mission.decisions == 0 and worker.requests == []
    mission.observe(snapshot(mission.last_time+.01))
    assert mission.advance(mission.last_time+.01, HOME, [.06, 0., 0.], 0., True) is None
    assert mission.decisions == 0
    requested(mission, worker)
    assert mission.decisions == 1 and mission.status == 'WAITING_NEXT'


def test_pending_model_is_nonblocking_and_does_not_credit_movement(pilot):
    mission, worker = pilot
    warmup(mission)
    requested(mission, worker)
    started = time.perf_counter()
    for _ in range(200):
        assert tick(mission) is None
    assert time.perf_counter()-started < .2
    assert mission.status == 'WAITING_NEXT'
    assert len(worker.requests) == 1 and len(mission.goals) == 1
    assert mission.history == []


def test_background_input_preparation_does_not_block_control(pilot):
    mission, worker = pilot
    release = threading.Event()
    original = mission._prepare_fn

    def slow(*args):
        assert release.wait(2.)
        return original(*args)

    mission._prepare_fn = slow
    warmup(mission)
    try:
        tick(mission)
        began = time.perf_counter()
        for _ in range(50):
            assert tick(mission) is None
        assert time.perf_counter()-began < .1
        assert not worker.requests
    finally:
        release.set()
    requested(mission, worker)


def test_left_home_sequence_uses_measured_history_and_independent_evaluation(pilot):
    mission, worker = pilot
    warmup(mission)
    first = activate(mission, worker)
    assert first.target_world == (0., 2., 2.)
    assert mission.history == []
    # Model activation and matching reference targets cannot complete a goal.
    mission.update(mission.last_time+.1, HOME, ZERO, 0.)
    assert mission.status == 'RUNNING'
    measured_arrival(mission, np.array(first.target_world))
    assert [r['candidate_id'] for r in mission.history] == ['LEFT']
    home = activate(mission, worker, 'HOME', 'RETURN', np.array(first.target_world))
    assert home.target_world == tuple(HOME)
    assert json.loads(worker.requests[-1]['prompt'])['history'][0]['candidate_id'] == 'LEFT'
    measured_arrival(mission, HOME)
    requested(mission, worker, 3)
    worker.answer('HOLD', 'REQUEST_STOP')
    until(mission, lambda: mission.status == 'COMPLETE')
    assert [r['candidate_id'] for r in mission.history] == ['LEFT', 'HOME']
    evaluated = evaluate_direction_task('left_return', HOME, 0., 2., mission.results,
                                       HOME, ZERO, True)
    assert evaluated['passed'] and evaluated['returned_after_required_visit']
    wrong_task = evaluate_direction_task('right_return', HOME, 0., 2., mission.results,
                                         HOME, ZERO, True)
    assert not wrong_task['passed']
    # A caller editing a retrieved event/result cannot rewrite the task ledger.
    copied = mission.results
    copied[1]['target_world'] = [100., 100., 100.]
    assert mission.results[1]['target_world'] == [0., 2., 2.]


def test_premature_model_stop_cannot_pass_task_evaluation(pilot):
    mission, worker = pilot
    warmup(mission)
    requested(mission, worker)
    worker.answer('HOLD', 'REQUEST_STOP')
    until(mission, lambda: mission.status == 'COMPLETE')
    assert mission.stop_requested and mission.history == []
    result = evaluate_direction_task('left_return', HOME, 0., 2., mission.results,
                                    HOME, ZERO, True)
    assert not result['passed'] and not result['required_visit_completed']


def test_hard_active_goal_failure_persists_and_cannot_be_resumed(pilot):
    mission, worker = pilot
    warmup(mission)
    activate(mission, worker)
    mission.update(mission.last_time+.1, HOME, ZERO, 0., 'tracking_position_limit')
    assert mission.status == 'FAILED' and mission.reason == 'tracking_position_limit'
    assert mission.results[-1]['status'] == 'FAILED'
    assert mission.results[-1]['reason'] == 'tracking_position_limit'
    assert any(e['kind'] == 'goal_failed' for e in mission.events)
    worker.answer('HOME', 'RETURN', request_id='obsolete_response')
    for _ in range(5):
        assert tick(mission) is None
        mission.update(mission.last_time+.01, HOME, ZERO, 0.)
    assert mission.status == 'FAILED' and len(mission.goals) == 2
    assert not mission.accepts(mission.current_goal.goal_id, mission.current_goal.revision)


def test_hard_failure_while_inference_pending_blocks_late_reply(pilot):
    mission, worker = pilot
    warmup(mission)
    requested(mission, worker)
    mission.update(mission.last_time+.01, HOME, ZERO, 0., 'uncertified_stop')
    worker.answer('LEFT')
    for _ in range(3):
        assert tick(mission) is None
    assert mission.status == 'FAILED' and mission.reason == 'uncertified_stop'
    assert len(mission.goals) == 1 and not any(e['kind'] == 'goal_activated' for e in mission.events)


def test_unknown_candidate_rejected_until_budget_exhausted(pilot):
    mission, worker = pilot
    warmup(mission)
    for index in range(3):
        requested(mission, worker, index+1)
        worker.answer('invented_coordinates_99')
        until(mission, lambda: mission.rejections == index+1)
    assert mission.status == 'FAILED' and mission.reason == 'policy_rejection_budget'
    assert len(mission.goals) == 1 and mission.history == []
    assert all(e['reason'] == 'UNKNOWN_CANDIDATE' for e in mission.events if e['kind'] == 'policy_rejected')


def test_decoded_decision_waits_for_fresh_map_and_revalidation(pilot):
    mission, worker = pilot
    warmup(mission)
    requested(mission, worker)
    worker.answer('LEFT')
    # Decode once, then let its originating observation expire. A slow semantic
    # decision is retained, but is not a current geometric authorization.
    tick(mission)
    assert 'ready_response' in mission.pending
    mission.observe(snapshot(mission.last_time, captured=time.perf_counter()-1.))
    for _ in range(5):
        assert tick(mission, fresh=False) is None
    assert mission.status == 'WAITING_NEXT' and mission.validation is None
    validated = []

    def validate(obs, position, yaw, selected):
        validated.append(obs['timestamp'])
        return True, obs, position

    mission._validate = validate
    until(mission, lambda: mission.status == 'RUNNING')
    assert validated and mission.events[-1]['kind'] == 'goal_activated'
    assert mission.events[-1]['revalidated_observation_time'] == validated[-1]
    assert len(worker.requests) == 1


def test_changed_geometry_rejects_previously_selected_target(pilot):
    mission, worker = pilot
    warmup(mission)
    requested(mission, worker)
    worker.answer('LEFT')
    mission._validate = lambda obs, position, yaw, selected: (False, obs, position)
    until(mission, lambda: mission.rejections == 1)
    assert mission.last_rejection['reason'] == 'CANDIDATE_NO_LONGER_OBSERVED'
    assert mission.status == 'WAITING_NEXT' and len(mission.goals) == 1


@pytest.mark.parametrize('changed', ['task', 'goal'])
def test_superseded_decoded_response_cannot_activate_after_validation(pilot, changed):
    mission, worker = pilot
    warmup(mission)
    requested(mission, worker)
    worker.answer('LEFT')
    tick(mission)
    assert 'ready_response' in mission.pending
    validation = Future()
    mission.validation = validation
    if changed == 'task':
        mission.task_revision += 1
    else:
        mission.goals[mission.index] = replace(mission.current_goal, revision=2)
    now = mission.last_time+.01
    checked = snapshot(now)
    mission.observe(checked)
    validation.set_result((True, checked, HOME.copy()))
    assert mission.advance(now, HOME, ZERO, 0., True) is None
    assert mission.last_rejection['reason'] == 'SUPERSEDED'
    assert mission.status == 'WAITING_NEXT' and len(mission.goals) == 1
    assert not any(e['kind'] == 'goal_activated' for e in mission.events)


def test_measured_return_without_requested_visit_and_unsettled_final_are_failures(pilot):
    mission, worker = pilot
    warmup(mission)
    activate(mission, worker, 'RIGHT')
    measured_arrival(mission, np.array([0., -2., 2.]))
    activate(mission, worker, 'HOME', 'RETURN', np.array([0., -2., 2.]))
    measured_arrival(mission, HOME)
    assert not evaluate_direction_task('left_return', HOME, 0., 2., mission.results,
                                       HOME, ZERO, True)['passed']
    assert not evaluate_direction_task('right_return', HOME, 0., 2., mission.results,
                                       HOME, [.1, 0., 0.], True)['passed']
    assert not evaluate_direction_task('right_return', HOME, 0., 2., mission.results,
                                       HOME, ZERO, False)['passed']


def compiled_plan(direction='LEFT'):
    return dict(steps=[dict(action='VISIT', direction=direction, distance_m=2.),
                       dict(action='RETURN_HOME'), dict(action='STOP')])


def test_compiled_clause_queue_advances_only_on_measured_goal_completion(pilot):
    mission, worker = pilot
    mission.task_plan = compiled_plan()
    assert mission.active_clause == dict(action='VISIT', direction='LEFT', distance_m=2.)
    warmup(mission)
    assert mission.clause_index == 0  # sensor warmup is not a language clause
    activated = activate(mission, worker)
    assert mission.clause_index == 0  # choosing a target is not completing it
    mission.update(mission.last_time+.1, HOME, ZERO, 0.)
    assert mission.clause_index == 0
    measured_arrival(mission, np.asarray(activated.target_world))
    assert mission.clause_index == 1 and mission.active_clause == dict(action='RETURN_HOME')
    activate(mission, worker, 'HOME', 'RETURN', np.asarray(activated.target_world))
    assert mission.clause_index == 1
    measured_arrival(mission, HOME)
    assert mission.clause_index == 2 and mission.active_clause == dict(action='STOP')
    requested(mission, worker, 3)
    worker.answer('HOLD', 'REQUEST_STOP')
    until(mission, lambda: mission.status == 'COMPLETE')
    assert evaluate_direction_task('left_return', HOME, 0., 2., mission.results,
                                   HOME, ZERO, True)['passed']


def test_wrong_direction_candidate_rejected_against_active_model_clause(pilot):
    mission, worker = pilot
    mission.task_plan = compiled_plan()
    warmup(mission)
    requested(mission, worker)
    worker.answer('RIGHT')
    until(mission, lambda: mission.rejections == 1)
    assert mission.last_rejection['reason'] == 'TASK_CLAUSE_MISMATCH'
    assert mission.clause_index == 0 and len(mission.goals) == 1
    assert mission.status == 'WAITING_NEXT' and mission.history == []


def test_compiled_task_rejects_premature_stop_before_any_required_visit(pilot):
    mission, worker = pilot
    mission.task_plan = compiled_plan()
    warmup(mission)
    requested(mission, worker)
    worker.answer('HOLD', 'REQUEST_STOP')
    until(mission, lambda: mission.rejections == 1)
    assert mission.last_rejection['reason'] == 'TASK_CLAUSE_NOT_COMPLETE'
    assert mission.status == 'WAITING_NEXT' and not mission.stop_requested
    assert mission.active_clause['direction'] == 'LEFT' and len(mission.goals) == 1


def test_model_compilation_error_is_not_silently_rewritten_from_instruction(pilot):
    mission, worker = pilot
    # Intentionally wrong model compilation: runtime obeys the supplied model
    # plan rather than silently using a hidden LEFT parser/evaluation answer.
    # The independent evaluator must therefore reject this completed episode.
    assert '左' in mission.instruction
    mission.task_plan = compiled_plan('RIGHT')
    warmup(mission)
    right = activate(mission, worker, 'RIGHT')
    assert right.target_world == (0., -2., 2.)
    measured_arrival(mission, np.asarray(right.target_world))
    activate(mission, worker, 'HOME', 'RETURN', np.asarray(right.target_world))
    measured_arrival(mission, HOME)
    requested(mission, worker, 3)
    worker.answer('HOLD', 'REQUEST_STOP')
    until(mission, lambda: mission.status == 'COMPLETE')
    evaluation = evaluate_direction_task('left_return', HOME, 0., 2., mission.results,
                                         HOME, ZERO, True)
    assert not evaluation['passed'] and not evaluation['required_visit_completed']
