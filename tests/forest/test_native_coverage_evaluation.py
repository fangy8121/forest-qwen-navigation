"""Independent geometry, discontinuity and model-provenance failure cases."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omni_drones.navigation.forest.native_coverage_evaluation import (
    audit_native_decisions, evaluate_native_coverage, native_evaluation_spec,
)


def scene(lengths=(10., 10., 10.), angle=0.):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    trees = [dict(row=i, position=(rotation@np.array([x, i*5.])).tolist(), ground_z=0.)
        for i, length in enumerate(lengths) for x in np.linspace(0., length, 6)]
    return dict(trees=trees)


def flight(points):
    pieces = [np.asarray(points[0], float)[None]]
    for a, b in zip(np.asarray(points[:-1], float), np.asarray(points[1:], float)):
        pieces.append(np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/.05))+1))[1:])
    actual = np.concatenate(pieces)
    actual = np.vstack([actual, np.tile(actual[-1], (15, 1))])
    times = np.arange(len(actual))*.1
    velocities = np.vstack([np.diff(actual, axis=0)/.1, np.zeros(3)])
    return actual, np.c_[times, actual, velocities]


def s_flight():
    return flight([[-1., 2.5, 2.5], [11., 2.5, 2.5], [11., 7.5, 2.5],
        [-1., 7.5, 2.5], [-1., 2.5, 2.5]])


def evaluate(actual, trace=None, **changes):
    arguments = dict(actual=actual, final_velocity=np.zeros(3), home=actual[0],
        physical_stop_verified=True, truth_margin_violation=False,
        mission=SimpleNamespace(status='COMPLETE', stop_requested=True,
            coverage_result='COMPLETE_UNDER_ASSUMPTIONS'), scene_info=scene(), trace=trace)
    arguments.update(changes)
    return evaluate_native_coverage(**arguments)


def test_complete_actual_s_route_and_one_second_measured_stop_pass():
    actual, trace = s_flight()
    result = evaluate(actual, trace)
    assert result['full_coverage_passed']
    assert result['true_corridor_count'] == 2
    assert result['s_pattern']['alternating_first_passes']
    assert all(c['uncovered_intervals'] == [] for c in result['truth_route_coverage'])


@pytest.mark.parametrize('bounds', [None, [0., 10., 0., 10.]])
def test_public_work_area_containing_entire_forest_preserves_full_route_requirement(bounds):
    actual, trace = s_flight()
    info = scene()
    info['evaluation_protocol'] = dict(work_area_bounds=bounds)
    result = evaluate(actual, trace, scene_info=info)
    assert result['full_coverage_passed']
    assert result['work_area_contains_entire_forest'] is (True if bounds is not None else None)
    assert result['coverage_scope'] == ('BOUND_PUBLIC_WORK_AREA' if bounds is not None else 'UNKNOWN_STAND_BOUNDARY')


def test_public_work_area_cutting_forest_cannot_shrink_true_denominator_or_pass():
    actual, trace = s_flight()
    baseline = evaluate(actual, trace)
    info = scene()
    info['evaluation_protocol'] = dict(work_area_bounds=[0., 9., 0., 10.])
    result = evaluate(actual, trace, scene_info=info)
    assert result['actual_full_route_coverage']
    assert result['evaluation_spec'] == baseline['evaluation_spec']
    assert result['truth_route_coverage'] == baseline['truth_route_coverage']
    assert result['public_work_area']['outside_tree_centers'] == 3
    assert result['work_area_contains_entire_forest'] is False
    assert not result['full_coverage_passed']


@pytest.mark.parametrize('bounds', [[0., 0., 0., 10.], [10., 0., 0., 10.],
    [0., 10., 0.], [0., float('inf'), 0., 10.], [0., 10., float('nan'), 10.]])
def test_malformed_public_work_area_cannot_pass(bounds):
    actual, trace = s_flight()
    info = scene()
    info['evaluation_protocol'] = dict(work_area_bounds=bounds)
    result = evaluate(actual, trace, scene_info=info)
    assert result['work_area_contains_entire_forest'] is False
    assert not result['full_coverage_passed']


def test_true_unequal_tail_remains_required_when_bilateral_region_was_flown():
    actual, trace = s_flight()
    result = evaluate(actual, trace, scene_info=scene((10., 10., 13.)))
    assert not result['actual_full_route_coverage']
    tail = result['truth_route_coverage'][1]
    assert tail['required_span'][1] == pytest.approx(13.)
    assert tail['bilateral_span'][1] == pytest.approx(10.)
    assert tail['uncovered_length_m'] > 1.
    assert result['false_completion']


def test_rotated_forest_has_same_coverage_and_s_behavior():
    angle = np.deg2rad(73.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    actual, trace = s_flight()
    actual[:, :2] = actual[:, :2]@rotation.T
    trace[:, 1:3] = actual[:, :2]
    trace[:, 4:6] = trace[:, 4:6]@rotation.T
    result = evaluate(actual, trace, scene_info=scene(angle=angle))
    assert result['full_coverage_passed']


def test_missing_corridor_cannot_be_replaced_by_return_over_first_corridor():
    actual, trace = flight([[-1., 2.5, 2.5], [11., 2.5, 2.5], [-1., 2.5, 2.5]])
    result = evaluate(actual, trace)
    assert result['truth_route_coverage'][0]['complete']
    assert not result['truth_route_coverage'][1]['complete']
    assert not result['full_coverage_passed']


def test_complete_parallel_forward_passes_do_not_count_as_s_shape():
    actual, trace = flight([[-1., 2.5, 2.5], [11., 2.5, 2.5], [11., -2., 2.5],
        [-1., -2., 2.5], [-1., 7.5, 2.5], [11., 7.5, 2.5], [11., -2., 2.5],
        [-1., -2., 2.5], [-1., 2.5, 2.5]])
    result = evaluate(actual, trace)
    assert result['actual_full_route_coverage']
    assert not result['s_pattern']['alternating_first_passes']
    assert not result['full_coverage_passed']


def corridor_sequence(order):
    points, end = [], -1.
    for corridor in order:
        y = 2.5+5.*corridor
        points.append([end, y, 2.5])
        end = 10.-end
        points.append([end, y, 2.5])
    # Return outside the tree-row spans after the final inspection passage.
    points.extend([[end, -2., 2.5], [-1., -2., 2.5], [-1., points[0][1], 2.5]])
    return flight(points)


def test_middle_start_covers_both_wings_with_adjacent_s_revisits():
    order = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    actual, trace = corridor_sequence(order)
    result = evaluate(actual, trace, scene_info=scene((10.,)*7))
    assert result['actual_full_route_coverage'] and result['full_coverage_passed']
    pattern = result['s_pattern']
    assert [p['index'] for p in pattern['passages'][:10]] == order
    assert pattern['adjacent_passages'] and pattern['alternating_passages']
    assert not pattern['adjacent_first_passes']  # C6 -> C1 first-visit jump is report-only.


def test_middle_start_headland_shortcut_is_not_a_sequence_of_adjacent_s_passes():
    actual, trace = corridor_sequence([1, 2, 3, 4, 5, 0])
    result = evaluate(actual, trace, scene_info=scene((10.,)*7))
    assert result['actual_full_route_coverage']
    assert not result['s_pattern']['adjacent_passages']
    assert result['s_pattern']['invalid_transitions']
    assert not result['full_coverage_passed']


def test_same_corridor_retraces_cannot_replace_other_corridors_or_excuse_a_jump():
    actual, trace = corridor_sequence([1, 1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
    result = evaluate(actual, trace, scene_info=scene((10.,)*7))
    assert result['full_coverage_passed']
    assert result['s_pattern']['inspection_passage_count'] == 12
    actual, trace = corridor_sequence([1, 1, 1])
    result = evaluate(actual, trace, scene_info=scene((10.,)*7))
    assert not result['actual_full_route_coverage']
    assert not result['s_pattern']['passed']


def test_out_and_back_inside_one_continuous_corridor_run_preserves_real_passages():
    actual, trace = flight([[.05, 2.5, 2.5], [9.95, 2.5, 2.5], [.05, 2.5, 2.5],
        [11., 2.5, 2.5], [11., 7.5, 2.5], [-1., 7.5, 2.5], [-1., 2.5, 2.5], [.05, 2.5, 2.5]])
    result = evaluate(actual, trace)
    assert result['full_coverage_passed']
    assert [p['direction'] for p in result['s_pattern']['passages'][:3]] == [1, -1, 1]


def test_endpoints_with_teleport_between_them_cannot_earn_route_coverage():
    actual = np.array([[-1., 2.5, 2.5], [10., 2.5, 2.5], [10., 7.5, 2.5],
        [0., 7.5, 2.5], [-1., 2.5, 2.5]])
    result = evaluate(actual)
    assert not result['actual_full_route_coverage']
    assert result['rejected_discontinuous_segments'] == 4


def test_timestamp_gap_cannot_credit_an_unobserved_interior_interval():
    actual, trace = s_flight()
    keep = ~((actual[:, 1] == 2.5) & (actual[:, 0] > 4.) & (actual[:, 0] < 5.))
    result = evaluate(actual[keep], trace[keep])
    assert result['truth_route_coverage'][0]['uncovered_length_m'] > .8
    assert not result['full_coverage_passed']


@pytest.mark.parametrize('change', ['missing_trace', 'moving_tail', 'failed_mission', 'margin_violation'])
def test_route_completion_alone_does_not_authorize_success(change):
    actual, trace = s_flight()
    changes = {}
    if change == 'missing_trace':
        trace = None
    elif change == 'moving_tail':
        trace[-10:, 4] = .06
    elif change == 'failed_mission':
        changes['mission'] = dict(status='FAILED', stop_requested=False,
            coverage_result='COMPLETE_UNDER_ASSUMPTIONS')
    else:
        changes['truth_margin_violation'] = True
    result = evaluate(actual, trace, **changes)
    assert result['actual_full_route_coverage']
    assert not result['full_coverage_passed']


def test_agl_uses_actual_triangle_terrain_and_does_not_extrapolate():
    actual, trace = s_flight()
    info = scene()
    xs, ys = np.arange(-2., 13.), np.arange(-2., 13.)
    info['evaluation_terrain'] = dict(xs=xs, ys=ys, heights=np.ones((len(xs), len(ys))))
    result = evaluate(actual, trace, scene_info=info)
    assert result['terrain_agl']['minimum_m'] == pytest.approx(1.5)
    assert result['terrain_agl']['maximum_m'] == pytest.approx(1.5)
    assert result['terrain_agl']['outside_terrain_samples'] == 0


@pytest.mark.parametrize('change', ['low_agl', 'missing_grid', 'outside_grid'])
def test_frozen_height_protocol_requires_valid_true_ground_and_height(change):
    actual, trace = s_flight()
    info = scene()
    info['evaluation_protocol'] = dict(agl_bounds=[1.5, 3.5], nominal_agl=2.5)
    xs, ys = np.arange(-2., 13.), np.arange(-2., 13.)
    info['evaluation_terrain'] = dict(xs=xs, ys=ys, heights=np.zeros((len(xs), len(ys))))
    if change == 'low_agl':
        info['evaluation_terrain']['heights'][:] = 2.
    elif change == 'missing_grid':
        del info['evaluation_terrain']
    else:
        info['evaluation_terrain']['xs'] += 3.
    result = evaluate(actual, trace, scene_info=info)
    assert result['actual_full_route_coverage']
    assert result['terrain_following_passed'] is False
    assert not result['full_coverage_passed']


def model_artifacts():
    decision = dict(candidate_id='B', behavior='CHANGE_AISLE')
    return dict(policy='qwen', model_metadata=dict(base_model='local/Qwen3.5-4B'),
        records=[dict(input=dict(request_id='decision_000', candidates=[
            dict(candidate_id='A', target_world=[1., 0., 2.5], corridor_id='C1', role='INSPECT'),
            dict(candidate_id='B', target_world=[0., 5., 2.5], corridor_id='C2', role='TRANSIT'),
            dict(candidate_id='HOLD', target_world=[0., 0., 2.5], role='HOLD')]),
            response=dict(request_id='decision_000', decision=decision,
                raw=json.dumps(decision), input_tokens=600, output_tokens=20, latency_seconds=.3))],
        events=[dict(kind='goal_activated', goal_id='qwen_002_B', request_id='decision_000',
            decision=decision, target_world=[0., 5., 2.5])])


def test_model_choice_is_bound_to_raw_reply_and_executed_target():
    result = audit_native_decisions(model_artifacts())
    assert result['passed']
    assert result['multi_option_decisions'] == 1
    assert result['multi_corridor_decisions'] == 1


def point_choice_artifacts():
    # point_choice_v1 (runs cc69e994, 63d4c129): the reply is a letter and
    # response.json is written before the supervisor maps it to a decision.
    data = model_artifacts()
    record = data['records'][0]
    record['input'].update(protocol='point_choice_v1', labels=dict(A='C', B='A', HOLD='B'))
    choice = dict(next_aisle=2, choice='A')
    record['response'] = dict(request_id='decision_000', response_schema='choice', choice=choice,
        raw=json.dumps(choice), input_tokens=900, output_tokens=15, latency_seconds=.5)
    return data


@pytest.mark.parametrize('change', ['valid', 'unknown_letter', 'worker_parse_differs',
                                    'duplicate_letter', 'other_candidate_executed'])
def test_point_choice_letter_is_bound_through_the_saved_labels(change):
    data = point_choice_artifacts()
    record = data['records'][0]
    if change == 'unknown_letter':
        record['response']['raw'] = json.dumps(dict(next_aisle=2, choice='D'))
        record['response']['choice']['choice'] = 'D'
    elif change == 'worker_parse_differs':
        record['response']['choice'] = dict(next_aisle=2, choice='C')
    elif change == 'duplicate_letter':
        record['input']['labels']['HOLD'] = 'A'
    elif change == 'other_candidate_executed':
        data['events'][0]['decision'] = dict(candidate_id='A', behavior='FOLLOW')
    result = audit_native_decisions(data)
    assert result['passed'] is (change == 'valid')
    if change == 'valid':
        assert result['records'][0]['decision'] == dict(candidate_id='B', behavior='CHANGE_AISLE')


def multi_order_artifacts():
    # Two rotated orders. Each reply leans to its letter A; summed per candidate
    # B wins (0.6+0.4 against 0.3+0.45), although the second reply alone picks A.
    data = point_choice_artifacts()
    record = data['records'][0]
    orders = [dict(labels=dict(B='A', A='B', HOLD='C')), dict(labels=dict(A='A', B='B', HOLD='C'))]
    record['input'].update(labels=orders[0]['labels'], orders=orders)
    replies = []
    for k, (letter, probabilities) in enumerate([('A', dict(A=.6, B=.3, C=.1)), ('A', dict(A=.45, B=.4, C=.15))]):
        choice = dict(next_aisle=2, choice=letter)
        replies.append(dict(request_id=f'decision_000_o{k}', response_schema='choice', choice=choice,
            raw=json.dumps(choice), letter_probabilities=probabilities, input_tokens=900, output_tokens=15,
            latency_seconds=.5))
    record['response'] = dict(request_id='decision_000', response_schema='choice', orders=replies,
        combined=dict(candidate_id='B'), input_tokens=1800, output_tokens=30, latency_seconds=1.)
    return data


@pytest.mark.parametrize('change', ['valid', 'combined_differs', 'missing_order', 'raw_differs',
                                    'order_request_id', 'no_inference'])
def test_a_choice_over_several_orders_is_rechecked_from_every_recorded_reply(change):
    data = multi_order_artifacts()
    response = data['records'][0]['response']
    if change == 'combined_differs':
        response['combined']['candidate_id'] = 'A'
    elif change == 'missing_order':
        response['orders'].pop()
    elif change == 'raw_differs':
        response['orders'][1]['raw'] = json.dumps(dict(next_aisle=2, choice='B'))
    elif change == 'order_request_id':
        response['orders'][1]['request_id'] = 'decision_000'
    elif change == 'no_inference':
        response['orders'][1]['output_tokens'] = 0
    result = audit_native_decisions(data)
    assert result['passed'] is (change == 'valid')
    if change == 'valid':
        assert result['records'][0]['decision'] == dict(candidate_id='B', behavior='CHANGE_AISLE')


def test_letter_probabilities_summed_per_candidate_cancel_a_position_bias():
    from omni_drones.navigation.forest.coverage_prompt import combined_choice
    orders = [dict(B='A', A='B', HOLD='C'), dict(A='A', B='B', HOLD='C')]
    biased = [dict(letter_probabilities=dict(A=.6, B=.3, C=.1)), dict(letter_probabilities=dict(A=.45, B=.4, C=.15))]
    pick, scores = combined_choice(list(zip(orders, biased)))
    assert pick == 'B' and scores['B'] == pytest.approx(1.) and scores['A'] == pytest.approx(.75)
    # Without probabilities each order's letter counts once; mass outside the
    # offered letters is ignored, and no offered mass at all gives no pick.
    assert combined_choice([(orders[0], dict(choice=dict(choice='B'))),
                            (orders[1], dict(letter_probabilities=dict(A=.9, Z=5.)))])[0] == 'A'
    assert combined_choice([(orders[0], dict(letter_probabilities=dict(Z=1.)))])[0] is None


def test_hold_wins_the_summed_scores_only_when_every_order_answered_it():
    # Run 8388ee83: one order moved (B 0.56), the other hovered (C 0.88); the sum
    # chose HOLD and the unchanged view repeated it until observation_no_progress.
    from omni_drones.navigation.forest.coverage_prompt import combined_choice
    orders = [dict(left='A', right='B', HOLD='C'), dict(right='A', left='B', HOLD='C')]
    split = [dict(choice=dict(choice='B'), letter_probabilities=dict(A=.437, B=.561, C=.002)),
             dict(choice=dict(choice='C'), letter_probabilities=dict(A=.034, B=.082, C=.883))]
    pick, scores = combined_choice(list(zip(orders, split)))
    assert pick == 'right' and max(scores, key=scores.get) == 'HOLD'
    both = [dict(choice=dict(choice='C'), letter_probabilities=dict(A=.1, B=.1, C=.8))]*2
    assert combined_choice(list(zip(orders, both)))[0] == 'HOLD'


def test_unrequested_observation_input_needs_no_reply_and_does_not_inflate_model_counts():
    data = model_artifacts()
    data['records'][0]['input']['model_requested'] = True
    data['records'].append(dict(input=dict(request_id='observation_001',
        model_requested=False, admission_reason='NO_SUPPORTED_MOVEMENT',
        candidates=[dict(candidate_id='HOLD', role='HOLD', target_world=[0., 0., 2.5])])))
    result = audit_native_decisions(data)
    assert result['passed']
    assert result['decisions'] == 1
    assert result['observation_only_inputs'] == 1
    assert [r['request_id'] for r in result['records']] == ['decision_000']


def inspection_artifacts():
    first = dict(candidate_id='A', role='INSPECT', corridor_id='C1', target_world=[1., 0., 2.5],
        commitment=dict(corridor_id='C1', direction=1))
    second = dict(candidate_id='B', role='TRANSIT', corridor_id='C1', target_world=[11., 0., 2.5],
        commitment=dict(corridor_id='C2', direction=-1), route=[
            dict(role='TRANSIT', corridor_id='C1', target_world=[11., 0., 2.5]),
            dict(role='TRANSIT', corridor_id='C2', target_world=[11., 5., 2.5]),
            dict(role='INSPECT', corridor_id='C2', target_world=[10., 5., 2.5])])
    data = dict(policy='qwen', model_metadata=dict(base_model='local/Qwen3.5-4B'), records=[], events=[])
    for index, candidate in enumerate((first, second)):
        request = f'decision_{index:03d}'
        decision = dict(candidate_id=candidate['candidate_id'],
            behavior='FOLLOW' if index == 0 else 'CHANGE_AISLE')
        data['records'].append(dict(input=dict(request_id=request, candidates=deepcopy([first, second])),
            response=dict(request_id=request, decision=decision, raw=json.dumps(decision),
                input_tokens=600, output_tokens=20, latency_seconds=.3)))
        data['events'].append(dict(kind='goal_activated', goal_id=f'branch_{index}', request_id=request,
            decision=decision, role=candidate['role'], corridor_id=candidate['corridor_id'],
            commitment=deepcopy(candidate['commitment']), target_world=candidate['target_world']))
    data['events'].append(dict(kind='coverage_continuation_activated', goal_id='first_C2_inspection',
        source_request_id='decision_001', parent_candidate_id='B', role='INSPECT', corridor_id='C2',
        commitment=deepcopy(second['commitment']), decision=dict(candidate_id='continue_4', behavior='FOLLOW'),
        target_world=[10., 5., 2.5]))
    return data


def test_multi_corridor_proof_accepts_transit_choice_with_explicit_later_inspection():
    result = audit_native_decisions(inspection_artifacts(), required_corridor_count=6)
    assert result['passed'] and result['corridor_selection_proven']
    assert result['model_selected_inspection_corridors'] == ['C1', 'C2']
    assert result['all_first_inspections_model_bound']


def test_bootstrap_choices_cannot_prove_model_selected_inspection_corridors():
    result = audit_native_decisions(model_artifacts(), required_corridor_count=2)
    assert result['artifact_bindings_passed']
    assert not result['passed'] and not result['corridor_selection_proven']


@pytest.mark.parametrize('change', ['singleton_branches', 'automatic_neighbor', 'wrong_parent'])
def test_corridor_proof_rejects_forced_branches_or_unselected_neighbor_inspection(change):
    data = inspection_artifacts()
    if change == 'singleton_branches':
        for record in data['records']:
            chosen = record['response']['decision']['candidate_id']
            record['input']['candidates'] = [c for c in record['input']['candidates'] if c['candidate_id'] == chosen]
    elif change == 'automatic_neighbor':
        data['events'][-1]['corridor_id'] = 'C3'
    else:
        data['events'][-1]['source_request_id'] = 'decision_000'
    assert not audit_native_decisions(data, required_corridor_count=2)['passed']


@pytest.mark.parametrize('change', ['valid', 'different_side', 'different_offset', 'inspection_identity'])
def test_outer_observation_extension_preserves_separate_outer_semantics(change):
    data = model_artifacts()
    chosen = data['records'][0]['input']['candidates'][1]
    commitment = dict(kind='outer_observation', corridor_id='C2', direction=1, side='high', offset_m=7.5)
    chosen.update(role='OBSERVE', corridor_id=None, target_world=[0., 7.5, 2.5], commitment=commitment)
    data['records'][0]['input']['state'] = dict(topology=[dict(corridor_id='C2', direction=[1., 0.], normal=[0., 1.])])
    decision = dict(candidate_id='B', behavior='FOLLOW')
    data['records'][0]['response'].update(decision=decision, raw=json.dumps(decision))
    data['events'][0].update(decision=decision, target_world=chosen['target_world'],
        role='OBSERVE', corridor_id=None, commitment=deepcopy(commitment))
    event = dict(kind='coverage_continuation_activated', goal_id='outer_continue',
        source_request_id='decision_000', parent_candidate_id='B', role='OBSERVE', corridor_id=None,
        commitment=deepcopy(commitment), decision=dict(candidate_id='continue_4', behavior='FOLLOW'),
        actual_position=[1., 7.5, 2.5], target_world=[4., 7.5, 2.5])
    if change == 'different_side':
        event['commitment']['side'] = 'low'
    elif change == 'different_offset':
        event['target_world'][1] = 8.
    elif change == 'inspection_identity':
        event['corridor_id'] = 'C2'
    data['events'].append(event)
    result = audit_native_decisions(data)
    assert result['passed'] is (change == 'valid')
    assert result['model_selected_inspection_corridors'] == []


@pytest.mark.parametrize('change', ['valid', 'different_corridor', 'reversed_direction', 'missing_parent'])
def test_routine_continuation_must_preserve_the_activated_model_intention(change):
    data = model_artifacts()
    chosen = data['records'][0]['input']['candidates'][1]
    chosen['commitment'] = dict(corridor_id='C2', direction=1)
    data['records'][0]['input']['state'] = dict(topology=[dict(corridor_id='C2', direction=[1., 0.])])
    event = dict(kind='coverage_continuation_activated', goal_id='coverage_continue_4',
        source_request_id='decision_000', parent_candidate_id='B',
        decision=dict(candidate_id='continue_4', behavior='FOLLOW'), target_world=[4., 5., 2.5],
        actual_position=[1., 5., 2.5], corridor_id='C2', commitment=dict(corridor_id='C2', direction=1))
    if change == 'different_corridor':
        event['corridor_id'] = 'C1'
    elif change == 'reversed_direction':
        event['target_world'] = [-2., 5., 2.5]
    elif change == 'missing_parent':
        data['events'] = []
    data['events'].append(event)
    assert audit_native_decisions(data)['passed'] is (change == 'valid')


def bounded_repair_artifacts(sign=1, explicit_scope=False):
    data = model_artifacts()
    d = np.array([np.cos(.7), np.sin(.7)])
    n = np.array([-d[1], d[0]])
    point = lambda s: np.r_[s*d+2.*n, 2.5].tolist()
    commitment = dict(corridor_id='C2', direction=sign, until_s=3.)
    if explicit_scope:
        commitment['scope'] = dict(kind='bounded_gap_repair', frame_direction=d.tolist(),
            endpoint_world_xy=point(3.)[:2], endpoint_s=3.)
    chosen = data['records'][0]['input']['candidates'][1]
    chosen.update(role='INSPECT', target_world=point(3.-2.*sign), commitment=commitment)
    data['records'][0]['input']['state'] = dict(topology=[dict(corridor_id='C2', direction=d.tolist())])
    decision = dict(candidate_id='B', behavior='FOLLOW')
    data['records'][0]['response'].update(decision=decision, raw=json.dumps(decision))
    data['events'][0].update(decision=decision, role='INSPECT', corridor_id='C2',
        commitment=deepcopy(commitment), target_world=chosen['target_world'])
    data['events'].append(dict(kind='coverage_continuation_activated', goal_id='repair_continue',
        source_request_id='decision_000', parent_candidate_id='B', role='INSPECT', corridor_id='C2',
        commitment=deepcopy(commitment), decision=dict(candidate_id='continue_4', behavior='FOLLOW'),
        actual_position=point(3.-sign), target_world=point(3.+.15*sign)))
    return data, d


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('explicit_scope', [False, True])
def test_bounded_repair_preserves_rotated_frame_and_arrival_tolerance(sign, explicit_scope):
    data, _ = bounded_repair_artifacts(sign, explicit_scope)
    result = audit_native_decisions(data)
    assert result['passed']
    assert result['bindings'][-1]['inspection_bound_respected']


@pytest.mark.parametrize('explicit_scope', [False, True])
@pytest.mark.parametrize('change', ['removed_bound', 'changed_bound', 'past_endpoint', 'far_target'])
def test_bounded_repair_rejects_removed_altered_or_exceeded_selected_endpoint(explicit_scope, change):
    data, d = bounded_repair_artifacts(explicit_scope=explicit_scope)
    event = data['events'][-1]
    if change == 'removed_bound':
        event['commitment'].pop('until_s')
    elif change == 'changed_bound':
        event['commitment']['until_s'] += 100.
    else:
        event['target_world'] = (np.asarray(event['target_world'])+
            np.r_[d*(.06 if change == 'past_endpoint' else 100.), 0.]).tolist()
    assert not audit_native_decisions(data)['passed']


@pytest.mark.parametrize('change', ['removed_commitment', 'changed_scope', 'missing_legacy_frame'])
def test_bound_cannot_disappear_from_branch_or_change_its_authorized_frame(change):
    data, _ = bounded_repair_artifacts(explicit_scope=change != 'missing_legacy_frame')
    if change == 'removed_commitment':
        data['events'][0].pop('commitment')
    elif change == 'changed_scope':
        data['events'][-1]['commitment']['scope']['frame_direction'] = [1., 0.]
    else:
        data['records'][0]['input']['state']['topology'] = []
    assert not audit_native_decisions(data)['passed']


def test_explicit_bound_uses_frozen_scope_even_when_observed_axis_changes():
    data, _ = bounded_repair_artifacts(explicit_scope=True)
    data['records'][0]['input']['state']['topology'][0]['direction'] = [1., 0.]
    assert audit_native_decisions(data)['passed']


def test_selected_transit_route_preserves_its_explicit_later_bounded_inspection():
    data = inspection_artifacts()
    scope = dict(kind='bounded_gap_repair', frame_direction=[1., 0.],
        endpoint_world_xy=[10., 5.], endpoint_s=10.)
    # Negative inspection ends at x=10; the selected headland route first
    # reaches x=11 in TRANSIT, which is a separate part of the same intention.
    for record in data['records']:
        record['input']['candidates'][1]['commitment'].update(until_s=10., scope=deepcopy(scope))
    for event in data['events'][1:]:
        event['commitment'].update(until_s=10., scope=deepcopy(scope))
    assert audit_native_decisions(data, required_corridor_count=2)['passed']


@pytest.mark.parametrize('change', ['rule', 'singleton', 'raw_changed', 'target_changed', 'behavior_changed', 'missing_metadata'])
def test_forced_rule_or_rewritten_model_decisions_cannot_pass_provenance(change):
    data = model_artifacts()
    if change == 'rule':
        data['records'][0]['response']['source'] = 'deterministic rule baseline, not a model'
    elif change == 'singleton':
        data['records'][0]['input']['candidates'].pop(0)
    elif change == 'raw_changed':
        data['records'][0]['response']['raw'] = json.dumps(dict(candidate_id='A', behavior='FOLLOW'))
    elif change == 'target_changed':
        data['events'][0]['target_world'] = [1., 0., 2.5]
    elif change == 'behavior_changed':
        data['events'][0]['decision'] = dict(candidate_id='B', behavior='FOLLOW')
    else:
        data['model_metadata'] = {}
    assert not audit_native_decisions(data)['passed']


def test_native_dataset_spec_contains_every_true_corridor_and_longer_tails():
    source = Path(__file__).resolve().parents[2]/'outputs/00_旧路线证据/01_森林巡检数据集/maps/train/000/map.json'
    if not source.exists():
        pytest.skip('local native dataset not present')
    original = json.loads(source.read_text())
    frozen = deepcopy(original)
    spec = native_evaluation_spec(original)
    assert len(spec['corridors']) == 6
    assert all(c['span'][0] <= c['bilateral_span'][0] and
        c['span'][1] >= c['bilateral_span'][1] for c in spec['corridors'])
    assert any(c['span'][1]-c['bilateral_span'][1] > 5. for c in spec['corridors'])
    assert original == frozen


def test_s_order_specification_requires_model_bound_inspections_not_an_aisle_choice():
    data = inspection_artifacts()
    for record in data['records']:
        chosen = record['response']['decision']['candidate_id']
        # Keep a non-corridor alternative so decisions stay genuinely multi-option.
        record['input']['candidates'] = [c for c in record['input']['candidates'] if c['candidate_id'] == chosen]+[
            dict(candidate_id='scan_left', role='OBSERVE', target_world=[0., 9., 2.5])]
    strict = audit_native_decisions(data, required_corridor_count=2)
    assert not strict['passed'] and not strict['corridor_selection_proven']
    constrained = audit_native_decisions(data, required_corridor_count=2, corridor_choice_required=False)
    assert constrained['passed'] and not constrained['corridor_selection_proven']
    assert not constrained['corridor_choice_required'] and 'S-order' in constrained['corridor_choice_scope']
    # Unbound first inspections still fail under the S-order specification.
    data['events'][-1]['source_request_id'] = 'decision_000'
    assert not audit_native_decisions(data, required_corridor_count=2, corridor_choice_required=False)['passed']


def commands(*rows):
    return [dict(command_id=f'c{i}', accepted=True, target=None, resume_after=False, **row) for i, row in enumerate(rows)]


def bound(command_rows):
    return SimpleNamespace(status='COMPLETE', stop_requested=True,
        coverage_result='COMPLETE_UNDER_ASSUMPTIONS', command_results=command_rows)


def test_subset_specification_requires_only_requested_aisles_numbered_from_home():
    # Aisles at y=2.5 and y=7.5; home at the y=2.5 side, so aisle 2 is y=7.5.
    actual, trace = flight([[-1., 2.5, 2.5], [-1., 7.5, 2.5], [11., 7.5, 2.5], [-1., 7.5, 2.5], [-1., 2.5, 2.5]])
    second = evaluate(actual, trace, mission=bound(commands(dict(intent='COVER', order='free', corridors=[2]))))
    assert second['full_coverage_passed'] and second['task_specification']['corridors'] == [2]
    assert [c['required_by_task'] for c in second['truth_route_coverage']] == [False, True]
    first = evaluate(actual, trace, mission=bound(commands(dict(intent='COVER', order='free', corridors=[1]))))
    assert not first['full_coverage_passed']


def test_home_on_the_far_side_reverses_aisle_numbers():
    actual, trace = flight([[-1., 7.5, 2.5], [11., 7.5, 2.5], [11., 7.5, 2.5], [-1., 7.5, 2.5]])
    result = evaluate(actual, trace, mission=bound(commands(dict(intent='COVER', order='free', corridors=[1]))))
    assert [c['lateral_index_from_home'] for c in result['truth_route_coverage']] == [2, 1]
    assert result['full_coverage_passed']


def test_s_pattern_is_required_only_without_temporary_commands():
    actual, trace = flight([[-1., 2.5, 2.5], [11., 2.5, 2.5], [11., -2., 2.5],
        [-1., -2., 2.5], [-1., 7.5, 2.5], [11., 7.5, 2.5], [11., -2., 2.5],
        [-1., -2., 2.5], [-1., 2.5, 2.5]])
    plain = evaluate(actual, trace, mission=bound(commands(dict(intent='COVER', order='S', corridors=None))))
    assert not plain['s_pattern']['passed'] and not plain['full_coverage_passed']
    visit = evaluate(actual, trace, mission=bound(commands(dict(intent='COVER', order='S', corridors=None),
        dict(intent='VISIT_CORRIDOR', order=None, corridors=None))))
    assert visit['full_coverage_passed'] and not visit['task_specification']['s_pattern_required']
