from copy import deepcopy
import pytest

from omni_drones.navigation.forest.coverage_command_audit import audit_coverage_commands


def received(identifier='pause', text='暂停', now=2., revision=2):
    return dict(kind='command_received', time=now, task_revision=revision,
        command_id=identifier, text=text, actual_speed_mps=.4,
        goal_id='old', revision=1)


def stopped(now=3.):
    return dict(event='measured_stop', timestamp=now, goal_id='old', revision=1,
        actual_velocity=[0., 0., 0.], actual_position=[1., 0., 2.], mode='handoff')


def test_planned_but_not_injected_command_is_missing_evidence():
    report = audit_coverage_commands([], [], [dict(command_id='pause', text='暂停')], policy='rule')
    assert report['status'] == 'MISSING_EVIDENCE'
    assert not report['complete']
    assert report['commands'][0]['checks'][0]['check'] == 'received'


def test_moving_pause_requires_actual_measured_stop_and_no_later_goal():
    command = received()
    plan = [dict(command_id='pause', text='暂停', min_speed=.1)]
    missing = audit_coverage_commands([command], [], plan, policy='rule')
    assert missing['status'] == 'MISSING_EVIDENCE'
    good = audit_coverage_commands([command], [stopped()], plan, policy='rule')
    assert good['status'] == 'COMPLETE'
    assert good['commands'][0]['checks'][0]['evidence']['received_while_moving']
    bad = audit_coverage_commands([command, dict(kind='goal_activated', time=4.,
        task_revision=2, goal_id='new', revision=2)], [stopped()], plan, policy='rule')
    assert bad['status'] == 'FAILED'


def test_wrong_corridor_binding_cannot_become_its_own_expected_answer():
    command = received('revisit', '回刚才那条通道补巡，然后继续巡检。', now=5.)
    events = [dict(kind='corridor_entered', time=1., corridor_id='R01_R02', previous_corridor=None),
        dict(kind='corridor_entered', time=4., corridor_id='R02_R03', previous_corridor='R01_R02'),
        command, dict(kind='command_bound', time=6., command_id='revisit',
            command=dict(intent='REVISIT_CORRIDOR', corridor_id='R02_R03', resume_after=True)),
        dict(kind='goal_activated', time=7., task_revision=2, goal_id='new', revision=2)]
    original = deepcopy(events)
    report = audit_coverage_commands(events, [stopped(5.5)], policy='rule')
    assert report['status'] == 'FAILED'
    row = report['commands'][0]
    assert row['frozen_target_corridor_id'] == 'R01_R02'
    assert next(c for c in row['checks'] if c['check'] == 'revisit_identity')['status'] == 'FAILED'
    assert events == original


def test_old_task_activation_after_replacement_fails_even_if_stop_exists():
    command = received('return', '返回起点，结束任务。')
    events = [command, dict(kind='command_bound', time=3., command_id='return',
        command=dict(intent='RETURN_HOME', corridor_id=None, resume_after=False)),
        dict(kind='goal_activated', time=4., task_revision=1, goal_id='stale', revision=2)]
    report = audit_coverage_commands(events, [stopped(3.5)], policy='rule')
    assert report['status'] == 'FAILED'
    assert any(c['check'] == 'no_superseded_activation' and c['status'] == 'FAILED' for c in report['global_checks'])


def test_revisit_needs_gap_clearance_and_restoration_evidence():
    command = received('revisit', '回刚才那条通道补巡，然后继续巡检。', now=5.)
    events = [dict(kind='corridor_entered', time=1., corridor_id='R01_R02', previous_corridor=None),
        dict(kind='corridor_entered', time=4., corridor_id='R02_R03', previous_corridor='R01_R02'), command,
        dict(kind='command_model_response', time=6., response=dict(source='deterministic rule baseline')),
        dict(kind='command_bound', time=6.1, command_id='revisit',
            command=dict(intent='REVISIT_CORRIDOR', corridor_id='R01_R02', resume_after=True)),
        dict(kind='goal_activated', time=7., task_revision=2, goal_id='new', revision=2),
        dict(kind='corridor_entered', time=8., corridor_id='R01_R02', previous_corridor='R02_R03'),
        dict(kind='revisit_completed', time=9., command_id='revisit', corridor_id='R01_R02', actual_position=[1., 0., 2.])]
    missing = audit_coverage_commands(events, [stopped(5.5)], policy='rule')
    assert missing['status'] == 'MISSING_EVIDENCE'
    events += [dict(kind='coverage_checkpoint', time=9.5, ledger=dict(corridors=[dict(
        corridor_id='R01_R02', uncovered=[], uncovered_length_m=0.)])),
        dict(kind='interrupted_task_restored', time=10., completed_temporary_corridor='R01_R02', restored_intent='COVER')]
    good = audit_coverage_commands(events, [stopped(5.5)], policy='rule')
    assert good['status'] == 'COMPLETE'
    assert good['policy_mode'] == 'RULE_BASELINE'
    assert good['commands'][0]['semantic_path'] == 'RULE_BASELINE'


def test_model_mode_requires_model_response_provenance():
    command = received('return', '返回起点，结束任务。')
    events = [command, dict(kind='command_bound', time=3., command_id='return',
        command=dict(intent='RETURN_HOME', corridor_id=None, resume_after=False)),
        dict(kind='goal_activated', time=4., task_revision=2, goal_id='new', revision=2)]
    report = audit_coverage_commands(events, [stopped(3.5)], policy='qwen_commands')
    assert report['status'] == 'MISSING_EVIDENCE'
    assert report['commands'][0]['semantic_path'] == 'UNVERIFIED'


def test_reusing_prior_stop_requires_continuous_rest_until_activation():
    command = received('resume', '继续巡检', now=2.)
    command['actual_speed_mps'] = 0.
    events = [command, dict(kind='command_bound', time=3., command_id='resume',
        command=dict(intent='RESUME', corridor_id=None, resume_after=False)),
        dict(kind='goal_activated', time=4., task_revision=2, goal_id='new', revision=2)]
    lower = [stopped(1.)]
    for i in range(1, 30):
        lower.append(dict(event='observation', timestamp=1.+i*.1, goal_id='old', revision=1,
            actual_position=[1., 0., 2.], actual_velocity=[0., 0., 0.], reference_velocity=[0., 0., 0.],
            actual_acceleration=[0., 0., 0.], reference_acceleration=[0., 0., 0.]))
    assert audit_coverage_commands(events, lower, policy='rule')['status'] == 'COMPLETE'
    outage = [e for e in lower if not 2. <= e.get('timestamp', 0.) <= 2.5]
    assert audit_coverage_commands(events, outage, policy='rule')['status'] == 'MISSING_EVIDENCE'


def test_history_binding_is_frozen_before_later_event_at_same_timestamp():
    command = received('revisit', '回刚才那条通道补巡，然后继续巡检。', now=5.)
    command['sequence'] = 2
    events = [dict(kind='corridor_entered', time=1., sequence=0, corridor_id='R01_R02', previous_corridor=None),
        dict(kind='corridor_entered', time=4., sequence=1, corridor_id='R02_R03', previous_corridor='R01_R02'),
        command, dict(kind='corridor_entered', time=5., sequence=3, corridor_id='R03_R04', previous_corridor='R02_R03'),
        dict(kind='command_bound', time=6., command_id='revisit',
            command=dict(intent='REVISIT_CORRIDOR', corridor_id='R01_R02', resume_after=True))]
    report = audit_coverage_commands(events, [stopped(5.5)], policy='rule')
    assert report['commands'][0]['frozen_target_corridor_id'] == 'R01_R02'


def _completion_case(kind):
    if kind == 'pause':
        return [received()], [stopped()], 3., 2.5
    if kind == 'resume':
        return [received('resume', '继续巡检'),
            dict(kind='command_bound', time=3.1, command_id='resume',
                command=dict(intent='RESUME', corridor_id=None, resume_after=False)),
            dict(kind='goal_activated', time=4., task_revision=2, goal_id='new', revision=2)], [stopped()], 4., 3.5
    temporary = kind == 'temporary_revisit'
    text = '回刚才那条通道补巡，然后继续巡检。' if temporary else '返回上一条通道'
    events = [dict(kind='corridor_entered', time=1., corridor_id='R01_R02', previous_corridor=None),
        dict(kind='corridor_entered', time=4., corridor_id='R02_R03', previous_corridor='R01_R02'),
        received('revisit', text, now=5.),
        dict(kind='command_model_response', time=6., response=dict(source='deterministic rule baseline')),
        dict(kind='command_bound', time=6.1, command_id='revisit',
            command=dict(intent='REVISIT_CORRIDOR', corridor_id='R01_R02', resume_after=temporary)),
        dict(kind='goal_activated', time=7., task_revision=2, goal_id='new', revision=2),
        dict(kind='corridor_entered', time=8., corridor_id='R01_R02', previous_corridor='R02_R03'),
        dict(kind='revisit_completed', time=9., command_id='revisit', corridor_id='R01_R02', actual_position=[1., 0., 2.])]
    if temporary:
        events += [dict(kind='coverage_checkpoint', time=9.5, ledger=dict(corridors=[dict(
            corridor_id='R01_R02', uncovered=[], uncovered_length_m=0.)])),
            dict(kind='interrupted_task_restored', time=10., completed_temporary_corridor='R01_R02', restored_intent='COVER')]
    return events, [stopped(5.5)], 10. if temporary else 9., 9.8 if temporary else 8.5


@pytest.mark.parametrize('kind', ['pause', 'resume', 'temporary_revisit', 'revisit'])
def test_unrelated_failure_after_proven_completion_is_not_attributed_to_command(kind):
    events, lower, completion_time, _ = _completion_case(kind)
    events.append(dict(kind='task_failed', time=20., reason='uncertified_stop'))
    report = audit_coverage_commands(events, lower, policy='rule')
    assert report['status'] == 'COMPLETE'
    row = report['commands'][0]
    assert row['completed_at'] == completion_time
    assert row['failure_attribution_until'] == completion_time
    assert row['subsequent_task_failures'][0]['time'] == 20.
    assert report['mission_task_failures'] == [dict(time=20., reason='uncertified_stop')]


@pytest.mark.parametrize('kind', ['pause', 'resume', 'temporary_revisit', 'revisit'])
def test_failure_before_command_completion_remains_failed(kind):
    events, lower, _, failure_time = _completion_case(kind)
    events.append(dict(kind='task_failed', time=failure_time, reason='uncertified_stop'))
    report = audit_coverage_commands(events, lower, policy='rule')
    assert report['status'] == 'FAILED'
    assert any(c['check'] == 'task_failure' and c['status'] == 'FAILED'
        for c in report['commands'][0]['checks'])


def test_binding_without_actual_resume_does_not_close_failure_window():
    events = [received('resume', '继续巡检'), dict(kind='command_bound', time=3., command_id='resume',
        command=dict(intent='RESUME', corridor_id=None, resume_after=False))]
    missing = audit_coverage_commands(events, [stopped()], policy='rule')
    assert missing['status'] == 'MISSING_EVIDENCE'
    assert missing['commands'][0]['completed_at'] is None
    events.append(dict(kind='task_failed', time=20., reason='uncertified_stop'))
    failed = audit_coverage_commands(events, [stopped()], policy='rule')
    assert failed['status'] == 'FAILED'
    assert failed['commands'][0]['completed_at'] is None
