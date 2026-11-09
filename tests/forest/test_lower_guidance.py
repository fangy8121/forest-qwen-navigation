import numpy as np
from omni_drones.navigation.forest.lower_guidance import gate_target, velocity_candidates
from omni_drones.navigation.forest.subgoal import Subgoal


def gate():
    return Subgoal('gate', 1, (10., 0., 3.), completion_mode='through',
        approach_direction=(1., 0., 0.), height_tolerance=.5, lateral_tolerance=.8, dwell_seconds=0.)


def test_height_misalignment_holds_before_plane_instead_of_escaping_to_next_anchor():
    target, holding = gate_target(gate(), [9., 0., 2.], [12., 0., 3.])
    assert holding and target[0] < 10.
    assert target[2] == 3.
    np.testing.assert_allclose(gate_target(gate(), [8., 0., 2.8], [12., 0., 3.])[0], [12., 0., 3.])
    target, holding = gate_target(gate(), [10.1, 0., 3.], [14., 0., 3.])
    assert holding and target[0] < 10.


def test_fine_height_reference_can_escape_coarse_grid_lock_but_retains_raw_fallback():
    choices = velocity_candidates([.5, 0., 0.], [0., 0., 2.], [2., 0., 2.3], .5)
    assert choices[0][2] > .1 and choices[0][0] > .3
    assert all(np.linalg.norm(v) <= .5+1e-12 for v in choices)
    assert any(np.array_equal(v, [.5, 0., 0.]) for v in choices)
    np.testing.assert_array_equal(choices[-1], np.zeros(3))
    np.testing.assert_array_equal(velocity_candidates([.5, 0., 0.], [0,0,2], [2,0,3], .5,
                                                    allow_motion=False), [np.zeros(3)])
