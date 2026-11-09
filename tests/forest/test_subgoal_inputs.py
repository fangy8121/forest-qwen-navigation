"""Observable anchors cannot cross obstacles, unknown gaps or public bounds."""
import numpy as np
import pytest
from PIL import Image

from omni_drones.navigation.forest.subgoal_inputs import (
    build_candidates, render_candidate_bev, validate_candidate,
)
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT


def snapshot(points=(), evidence=None):
    if evidence is None:
        x, y, z = np.meshgrid(np.arange(-4., 4.01, .1), np.arange(-4., 4.01, .1),
                              np.arange(-.3, .301, .1))
        evidence = np.c_[x.ravel(), y.ravel(), z.ravel()]
    return dict(points=np.asarray(points, float).reshape(-1, 3),
                evidence=np.asarray(evidence, float).reshape(-1, 3), timestamp=3.)


def by_id(candidates):
    return {c['id']: c for c in candidates}


def test_observed_open_scene_has_both_language_choices_without_task_input():
    result = by_id(build_candidates(snapshot(), np.zeros(3), 0., np.zeros(3), DEFAULT_FLIGHT))
    assert {'HOLD', 'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'} <= result.keys()
    np.testing.assert_allclose(result['LEFT']['target_world'], [0., 2., 0.])
    np.testing.assert_allclose(result['RIGHT']['target_world'], [0., -2., 0.])
    assert 'HOME' not in result
    assert result['HOLD']['movement_authorized'] is False
    assert all(c['execution_certificate'] is False for c in result.values())


def test_endpoint_clear_does_not_authorize_segment_crossing_obstacle():
    candidates = by_id(build_candidates(snapshot([[1., 0., 0.]]), np.zeros(3), 0.,
                                       [-2., 0., 0.], DEFAULT_FLIGHT))
    assert 'FORWARD' not in candidates
    assert 'LEFT' in candidates and 'RIGHT' in candidates


def test_selected_candidate_revalidation_rejects_new_obstacle_and_unknown_gap():
    initial = snapshot()
    candidate = by_id(build_candidates(initial, np.zeros(3), 0., np.zeros(3), DEFAULT_FLIGHT))['FORWARD']
    assert validate_candidate(initial, np.zeros(3), 0., candidate, DEFAULT_FLIGHT)
    assert not validate_candidate(snapshot([[1., 0., 0.]]), np.zeros(3), 0., candidate, DEFAULT_FLIGHT)
    initial['evidence'] = initial['evidence'][np.abs(initial['evidence'][:, 0]-1.) > .4]
    assert not validate_candidate(initial, np.zeros(3), 0., candidate, DEFAULT_FLIGHT)


def test_selected_candidate_revalidation_respects_public_bounds():
    snap = snapshot()
    candidate = dict(id='chosen', target_world=[2., 0., 0.])
    assert validate_candidate(snap, np.zeros(3), 0., candidate, DEFAULT_FLIGHT,
                              operating_bounds=(-1., 3., -1., 1.), height_bounds=(-.1, .1))
    assert not validate_candidate(snap, np.zeros(3), 0., candidate, DEFAULT_FLIGHT,
                                  operating_bounds=(-1., 1., -1., 1.))
    assert not validate_candidate(snap, np.zeros(3), 0., candidate, DEFAULT_FLIGHT,
                                  height_bounds=(1., 2.))


@pytest.mark.parametrize('candidate', [
    dict(id='HOLD', target_world=[0., 0., 0.]),
    dict(candidate_id='HOLD', target_world=[2., 0., 0.]),
    dict(kind='hold', target_world=[2., 0., 0.]),
    dict(target_world=[2., 0., 0.], movement_authorized=False),
    dict(target_world=[float('nan'), 0., 0.]),
    dict(target_world=[float('inf'), 0., 0.]),
    dict(target_world=[1e100, 0., 0.]), dict(), None,
])
def test_selected_hold_or_malformed_candidate_is_not_movement(candidate):
    assert not validate_candidate(snapshot(), np.zeros(3), 0., candidate, DEFAULT_FLIGHT)


def test_unknown_gap_and_previously_visited_anchor_remain_unavailable():
    snap = snapshot()
    snap['evidence'] = snap['evidence'][np.abs(snap['evidence'][:, 0]-1.) > .4]
    candidates = by_id(build_candidates(snap, np.zeros(3), 0., [0., 0., 0.], DEFAULT_FLIGHT,
                                       visited_anchors=[dict(id='old', target_world=[2.5, 0., 0.])]))
    assert 'FORWARD' not in candidates and 'HOME' not in candidates and 'VISIT_old' not in candidates
    assert 'LEFT' in candidates
    returning = by_id(build_candidates(snap, [2., 0., 0.], np.pi, np.zeros(3), DEFAULT_FLIGHT))
    assert 'HOME' not in returning


def test_empty_observation_only_holds_and_does_not_certify_home():
    candidates = build_candidates(snapshot(evidence=[]), np.zeros(3), 0., np.zeros(3), DEFAULT_FLIGHT)
    assert [c['id'] for c in candidates] == ['HOLD']


def test_current_heading_does_not_rotate_stable_takeoff_anchors():
    result = by_id(build_candidates(snapshot(), np.zeros(3), np.pi/2, np.zeros(3), DEFAULT_FLIGHT))
    np.testing.assert_allclose(result['FORWARD']['target_world'], [2., 0., 0.], atol=1e-6)
    np.testing.assert_allclose(result['LEFT']['target_world'], [0., 2., 0.], atol=1e-6)
    np.testing.assert_allclose(result['LEFT']['relative_body_xyz_m'], [2., 0., 0.], atol=1e-6)
    np.testing.assert_allclose(result['LEFT']['relative_home_xyz_m'], [0., 2., 0.], atol=1e-6)


def test_takeoff_heading_defines_ring_and_arrived_anchor_is_not_reoffered():
    result = by_id(build_candidates(snapshot(), [0., 2., 0.], np.pi/2, np.zeros(3), DEFAULT_FLIGHT,
                                   anchor_yaw=np.pi/2))
    assert 'FORWARD' not in result and 'HOME' in result
    np.testing.assert_allclose(result['LEFT']['target_world'], [-2., 0., 0.], atol=1e-6)


def test_public_bounds_remove_outside_anchors():
    result = by_id(build_candidates(snapshot(), np.zeros(3), 0., np.zeros(3), DEFAULT_FLIGHT,
                                   operating_bounds=(-1., 3., -3., 3.), height_bounds=(-.1, .1)))
    assert 'BACKWARD' not in result and 'FORWARD' in result
    result = build_candidates(snapshot(), np.zeros(3), 0., np.zeros(3), DEFAULT_FLIGHT,
                              height_bounds=(1., 2.))
    assert [c['id'] for c in result] == ['HOLD']


def test_only_observation_fields_affect_candidates():
    snap = snapshot()
    expected = build_candidates(snap, np.zeros(3), 0., np.zeros(3), DEFAULT_FLIGHT)
    snap.update(teacher_route=[[80., 80., 80.]], true_lane=9, instruction='go right')
    assert build_candidates(snap, np.zeros(3), 0., np.zeros(3), DEFAULT_FLIGHT) == expected


def test_compact_render_has_valid_dimensions_and_unknown_legend(tmp_path):
    snap = snapshot(evidence=[])
    candidates = build_candidates(snap, np.zeros(3), 0., np.zeros(3), DEFAULT_FLIGHT)
    path = tmp_path/'candidate.png'
    result = render_candidate_bev(snap, np.zeros(3), 0., np.zeros(3), candidates, path,
                                  DEFAULT_FLIGHT, trail=[[0., 0., 0.]])
    assert result['unknown_cells'] > 0 and result['known_clear_cells'] == 0
    with Image.open(path) as image:
        assert image.size == (860, 600)


@pytest.mark.parametrize('changes', [dict(yaw=float('nan')), dict(step_distance=0.),
                                    dict(position=[0., 0., float('inf')])])
def test_nonfinite_or_invalid_inputs_are_rejected(changes):
    args = dict(snapshot=snapshot(), position=np.zeros(3), yaw=0., home_world=np.zeros(3), flight=DEFAULT_FLIGHT)
    args.update(changes)
    with pytest.raises(ValueError):
        build_candidates(**args)
