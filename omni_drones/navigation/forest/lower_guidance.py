"""Public goal guidance for moving gates and sub-voxel height references.

These hints never authorize motion: every velocity remains subject to the same
observed-body, obstacle and actual-state stopping checks in lower_flight.
"""
import numpy as np


def gate_target(goal, position, nominal_target, lookahead=2.):
    """Hold before a misaligned through gate; never silently skip its plane."""
    position, nominal = np.asarray(position, float), np.asarray(nominal_target, float)
    if goal.completion_mode != 'through':
        return nominal.copy(), False
    direction = np.asarray(goal.approach_direction)
    lateral = np.array([-direction[1], direction[0], 0.])
    delta = position-goal.target_world
    signed = float(delta@direction)
    lateral_limit = goal.lateral_tolerance or goal.position_tolerance
    height_limit = goal.height_tolerance or goal.position_tolerance
    # Reserve a little of the task's unchanged tolerance for measured tracking.
    ready = (abs(delta@lateral) < .8*lateral_limit and abs(delta[2]) < .8*height_limit)
    if signed >= 0. or (signed > -lookahead and not ready):
        return np.asarray(goal.target_world)-max(.6, goal.position_tolerance+.3)*direction, True
    return nominal.copy(), False


def velocity_candidates(raw, position, target, speed_limit, *, allow_motion=True,
                        height_gain=.6, vertical_limit=.2):
    """Offer a bounded fine-height reference before coarse A* velocity fallbacks.

    Coarse 0.4m search nodes can keep all future z equal to current z despite a
    gradual rising public target. A small continuous height component lets the
    execution certificate test supported motion without marking upper voxels
    free. Raw A* alternatives stay available when that component is rejected.
    """
    raw, position, target = map(lambda x: np.asarray(x, float), (raw, position, target))
    if not allow_motion:
        return [np.zeros(3)]
    choices = []
    height_error = float(target[2]-position[2])
    if abs(height_error) > .05:
        fine = raw.copy()
        fine[2] = np.clip(height_gain*height_error, -vertical_limit, vertical_limit)
        fine *= min(1., speed_limit/max(np.linalg.norm(fine), 1e-12))
        choices.append(fine)
    choices.append(raw)
    result = []
    for choice in choices:
        for scale in (1., .5, .25):
            value = scale*choice
            if np.linalg.norm(value) > 1e-8 and not any(np.allclose(value, old) for old in result):
                result.append(value)
    return result+[np.zeros(3)]
