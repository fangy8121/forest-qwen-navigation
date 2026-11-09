import numpy as np

from omni_drones.navigation.forest.geometry import INFLATION, HEIGHT_LEVELS
from omni_drones.navigation.forest.observation import (
    History, SampleSelector, grid_observation, representation_spec, render_observation)


def frame(hits, position=(0., 0., 0.), quaternion=(1., 0., 0., 0.)):
    hits = np.asarray(hits, dtype=float)
    lengths = np.linalg.norm(hits, axis=1)
    return dict(timestamp=0., points_sensor=hits, valid=np.ones(len(hits), bool),
                sensor_position=np.asarray(position), sensor_quaternion=np.asarray(quaternion),
                directions_world=hits/lengths[:, None], distances=lengths, max_range=6.)


def test_history_retains_both_heights_and_stops_free_rays_at_hits():
    history = History()
    history.add(frame([[2., 0., 1.], [2., 0., -1.]]))
    points, evidence = history.snapshot(0., np.zeros(3), 0.)
    np.testing.assert_allclose(points, [[2., 0., 1.], [2., 0., -1.]])
    assert evidence.shape[1] == 3
    assert set(np.sign(evidence[:, 2])) == {-1., 0., 1.}
    assert evidence[:, 0].max() < 2.
    assert history.snapshot(0., np.zeros(3), 0., include_evidence=False)[1].shape == (0, 3)


def test_snapshot_applies_sensor_rotation_before_gravity_aligned_yaw():
    history = History()
    f = frame([[2., 0., 0.]], position=(1., 2., 3.),
              quaternion=(np.sqrt(.5), 0., -np.sqrt(.5), 0.))
    history.add(f)
    points, _ = history.snapshot(0., [1., 2., 4.], .6, include_evidence=False)
    np.testing.assert_allclose(points, [[0., 0., 1.]], atol=1e-12)
    assert points[0, 2] == 1.


def test_high_branch_and_low_barrier_have_different_height_slices():
    for obstacle_z in (-.5, 1.):
        points = np.array([[2., 0., obstacle_z]])
        masks = []
        for height in HEIGHT_LEVELS:
            xs, ys, distance, _ = grid_observation(points, np.empty((0, 3)), height=height)
            ix, iy = np.argmin(abs(xs-2.)), np.argmin(abs(ys))
            masks.append(distance[ix, iy] <= INFLATION)
        assert any(masks)
        assert not masks[list(HEIGHT_LEVELS).index(0.)]
        assert all(height*obstacle_z > 0 for height, blocked in zip(HEIGHT_LEVELS, masks) if blocked)


def test_single_plane_evidence_is_local_to_its_height_and_tilt_bound():
    axis = np.arange(-1., 1.01, .2)
    x, y = np.meshgrid(axis, axis)
    evidence = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    xs, ys, _, seen = grid_observation(np.empty((0, 3)), evidence, max_acceleration=1.)
    assert seen[np.argmin(abs(xs)), np.argmin(abs(ys))]
    for height in (-.75, .75):
        assert not grid_observation(np.empty((0, 3)), evidence, height=height)[3].any()
    assert not grid_observation(np.empty((0, 3)), evidence, max_acceleration=3.5)[3].any()
    volume = np.concatenate([evidence + [0., 0., z] for z in np.arange(-.4, .41, .2)])
    xs, ys, _, seen = grid_observation(np.empty((0, 3)), volume)
    assert seen[np.argmin(abs(xs)), np.argmin(abs(ys))]


def test_sample_selector_distinguishes_vertical_scene_and_goal():
    selector = SampleSelector()
    evidence = np.empty((0, 3))
    first = selector.describe(np.array([[2., 0., 1.]]), evidence, [6., 0., 0.], [0., 0., 2.], 0.)
    changed = selector.describe(np.array([[2., 0., -1.]]), evidence, [6., 0., 0.], [0., 0., 2.], 0.)
    assert selector.change(first['signature'], changed['signature']) > .5
    selector.accept(first)
    assert selector.reason(changed) is None
    changed_goal = selector.describe(np.array([[2., 0., 1.]]), evidence, [6., 0., 1.], [0., 0., 2.], 0.)
    assert selector.reason(changed_goal) is None


def test_render_equal_slices_with_xyz_debug_path(tmp_path):
    from PIL import Image
    output = tmp_path/'slices.png'
    render_observation(np.array([[2., 0., 1.], [4., 0., -1.]]), np.empty((0, 3)),
                       np.array([6., 0., .5]), output, path=[[0., 0., 0.], [3., 0., -.5], [6., 0., .5]])
    assert Image.open(output).size == (500*len(HEIGHT_LEVELS), 500)
    assert representation_spec()['height_levels'] == list(HEIGHT_LEVELS)


def test_thin_obstacle_between_display_heights_is_never_hidden():
    from omni_drones.navigation.forest.geometry import VERTICAL_RADIUS
    for obstacle_z in [.5, -.5, *np.linspace(min(HEIGHT_LEVELS)-VERTICAL_RADIUS, max(HEIGHT_LEVELS)+VERTICAL_RADIUS, 61)]:
        assert min(abs(obstacle_z-height) for height in HEIGHT_LEVELS) <= VERTICAL_RADIUS+1e-12
    points = np.array([[2., 0., .5]])
    assert any((grid_observation(points, np.empty((0, 3)), height=height)[2] <= INFLATION).any()
               for height in HEIGHT_LEVELS)


def test_lower_obstacles_remain_in_history_and_safety_outside_main_slices():
    from omni_drones.navigation.forest.geometry import parse_path
    from omni_drones.navigation.forest.observation import lower_obstacles
    from omni_drones.navigation.forest.planning import PathSafety
    from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT
    import json
    history = History()
    history.add(frame([[3., 0., -1.4]]))
    points, evidence = history.snapshot(0., np.zeros(3), 0.)
    np.testing.assert_allclose(lower_obstacles(points), [[3., 0., -1.4]])
    assert all(not (grid_observation(points, evidence, height=h)[2] <= INFLATION).any()
               for h in HEIGHT_LEVELS)
    assert not PathSafety(points, evidence, None, DEFAULT_FLIGHT).clear_points([[3., 0., -1.3]])[0]
    parsed = parse_path(json.dumps(dict(status='GO', points=[[i, 0., -1.3] for i in range(1, 7)])))
    assert parsed['points'][0][2] == -1.3


def test_sample_selector_keeps_lower_obstacle_height_changes():
    selector = SampleSelector()
    args = (np.empty((0, 3)), [6., 0., 0.], [0., 0., 2.], 0.)
    first = selector.describe(np.array([[2., 0., -1.4]]), *args)
    changed = selector.describe(np.array([[2., 0., -1.6]]), *args)
    np.testing.assert_array_equal(first['signature'][:-1], changed['signature'][:-1])
    assert selector.change(first['signature'], changed['signature']) == 1.
    selector.accept(first)
    assert selector.reason(changed) is None


def test_prompt_separates_display_heights_from_planning_range():
    from omni_drones.navigation.forest.qwen import task_prompt
    prompt = task_prompt(np.array([6., 0., -1.3]))
    assert 'z=[-0.5, 0.0, 0.5, 1.0, 1.5]' in prompt
    assert 'z范围=[-1.5, 1.5]' in prompt
    assert 'Lower hits' in prompt
    assert '不表示自由空间' in prompt
