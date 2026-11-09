"""Arrival, deadline, heading and physical diagnostic geometry regressions."""
from dataclasses import replace
import numpy as np
import pytest
from omni_drones.navigation.forest.subgoal import Subgoal, GoalMonitor, HeadingReference, angle_error
from omni_drones.navigation.forest.pilot_scene import cases, make_scene
from omni_drones.navigation.forest.evaluation import truth_collision, sample_polyline
from omni_drones.navigation.forest.geometry import INFLATION, VERTICAL_RADIUS


def test_pass_through_wrong_heading_and_motion_do_not_complete():
    goal = Subgoal('left', 1, (0., 2., 2.), arrival_yaw=np.pi/2)
    monitor = GoalMonitor(goal)
    assert monitor.update(0., goal.target_world, [0., .6, 0.], np.pi/2)['status'] == 'RUNNING'
    assert monitor.update(1., goal.target_world, [0., 0., 0.], 0.)['status'] == 'RUNNING'
    assert monitor.update(2., goal.target_world, [0., 0., 0.], np.pi/2)['status'] == 'RUNNING'
    assert monitor.update(2.4, [0., 2.5, 2.], [0., 0., 0.], np.pi/2)['dwell_seconds'] == 0.
    assert monitor.update(3., goal.target_world, [0., 0., 0.], np.pi/2)['status'] == 'RUNNING'
    assert monitor.update(3.51, goal.target_world, [0., 0., 0.], np.pi/2)['status'] == 'COMPLETE'


def test_failure_takes_priority_over_completion():
    goal = Subgoal('goal', 1, (0., 0., 2.))
    monitor = GoalMonitor(goal)
    monitor.update(0., goal.target_world, [0., 0., 0.], 0.)
    assert monitor.update(1., goal.target_world, [0., 0., 0.], 0., 'collision')['reason'] == 'collision'
    assert monitor.update(2., goal.target_world, [0., 0., 0.], 0.)['status'] == 'FAILED'


def test_motion_in_a_detour_resets_stall_but_does_not_disable_deadline():
    goal = Subgoal('goal', 1, (5., 0., 2.), timeout_seconds=5., no_motion_seconds=2.)
    monitor = GoalMonitor(goal)
    for t in range(5):
        assert monitor.update(float(t), [-float(t), 0., 2.], [-1., 0., 0.], 0.)['status'] == 'RUNNING'
    assert monitor.update(5., [-5., 0., 2.], [-1., 0., 0.], 0.)['reason'] == 'timeout'
    monitor = GoalMonitor(goal)
    monitor.update(0., [0., 0., 2.], [0., 0., 0.], 0.)
    assert monitor.update(2., [0., 0., 2.], [0., 0., 0.], 0.)['reason'] == 'no_motion'


def test_invalid_goal_and_nonmonotonic_state_are_rejected():
    goal = Subgoal('goal', 1, (1., 0., 2.))
    for change in ({'revision': 0}, {'target_world': (np.nan, 0., 2.)},
                   {'position_tolerance': 0.}, {'arrival_yaw': np.inf}):
        with pytest.raises(ValueError):
            replace(goal, **change)
    monitor = GoalMonitor(goal)
    monitor.update(1., [0., 0., 2.], [0., 0., 0.], 0.)
    with pytest.raises(ValueError):
        monitor.update(.5, [0., 0., 2.], [0., 0., 0.], 0.)


@pytest.mark.parametrize('start,target', [(0., np.pi/2), (0., -np.pi/2),
                                        (np.pi-.05, -np.pi+.05)])
def test_heading_converges_and_respects_limits_across_wrap(start, target):
    reference = HeadingReference(start, 1., 2.)
    values = np.array([reference.step(target, .01) for _ in range(700)])
    assert np.abs(values[:, 1]).max() <= 1.+1e-12
    assert np.abs(values[:, 2]).max() <= 2.+1e-12
    assert np.max(np.abs(np.diff(values[:, 1]))/.01) <= 2.+1e-10
    assert abs(angle_error(target, values[-1, 0])) < 1e-3
    assert abs(values[-1, 1]) < 1e-3
    if start > 3.:
        assert np.max(np.abs(values[:, 0]-start)) < .2


def test_pilot_has_clear_goal_regions_but_requires_a_bypass():
    info, vertices, faces, _ = make_scene()
    truth = dict(mesh=(vertices, faces))
    for goal in cases():
        assert not truth_collision(np.array([info['start'], goal.target_world]), truth, INFLATION, VERTICAL_RADIUS)
    direct = sample_polyline(np.array([info['start'], cases()[-1].target_world]))
    assert truth_collision(direct, truth, INFLATION, VERTICAL_RADIUS)
    detour = sample_polyline(np.array([info['start'], [1., 1.5, 2.], [4.5, 1.5, 2.], cases()[-1].target_world]))
    assert not truth_collision(detour, truth, INFLATION, VERTICAL_RADIUS)


def test_shuttling_without_getting_closer_ends_as_no_progress():
    # Run a0a40f7f: 0.4 m back-and-forth in a dead-end pocket kept resetting
    # no_motion for 200 s. Progress is distance to the goal, not motion.
    goal = Subgoal('goal', 1, (5., 0., 2.), timeout_seconds=100., no_motion_seconds=2., no_progress_seconds=10.)
    monitor = GoalMonitor(goal)
    for step in range(100):
        x = .4*(step % 2)
        feedback = monitor.update(.1*step+.05*step, [x, 0., 2.], [.3, 0., 0.], 0.)
        if feedback['status'] != 'RUNNING':
            break
    assert feedback['reason'] == 'no_progress' and 10. <= .15*step < 11.
    # Real approach resets the window; slow but steady progress is not stopped.
    monitor = GoalMonitor(goal)
    for t in range(40):
        assert monitor.update(float(t), [.12*t, 0., 2.], [.12, 0., 0.], 0.)['status'] == 'RUNNING'
    # Disabled by default; near the goal it never fires.
    assert GoalMonitor(replace(goal, no_progress_seconds=None)).update(30., [0., 0., 2.], [.3, 0., 0.], 0.)['status'] == 'RUNNING'
    near = GoalMonitor(goal)
    near.update(0., [4.6, 0., 2.], [.3, 0., 0.], 0.)
    assert near.update(20., [4.62, .05, 2.], [.3, 0., 0.], .1)['status'] == 'RUNNING'
    with pytest.raises(ValueError, match='no_progress'):
        Subgoal('goal', 1, (5., 0., 2.), no_progress_seconds=0.)
