"""Check bounded Lee tracking requests through physical rotor force equilibrium."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.spatial.transform import Rotation

from omni_drones.controllers.lee_position_controller import LeePositionController


def controller():
    params = yaml.safe_load(Path('omni_drones/robots/assets/usd/iris.yaml').read_text())
    return LeePositionController(9.81, params)


def equilibrium_state(acceleration):
    """Level-heading body orientation whose thrust balances gravity plus a."""
    force = np.asarray(acceleration, dtype=float)+[0., 0., 9.81]
    z = force/np.linalg.norm(force)
    y = np.cross(z, [1., 0., 0.]); y /= np.linalg.norm(y)
    rotation = np.column_stack([np.cross(y, z), y, z])
    q = Rotation.from_matrix(rotation).as_quat()  # xyzw -> wxyz
    state = torch.zeros(1, 13)
    state[0, 3:7] = torch.tensor(q[[3, 0, 1, 2]], dtype=torch.float32)
    return state, float(np.linalg.norm(force))


def rotor_wrench(control, commands):
    forces = (commands+1.)*.5*control.max_thrusts
    return torch.linalg.solve(control.mixer, forces.T).T


def test_combined_feedback_and_feedforward_use_one_vector_norm_budget():
    control = controller()
    limit = .8
    # Position feedback (6), velocity feedback (4.7), and feedforward (2)
    # all request acceleration in each axis. A component-wise clamp would
    # produce sqrt(3) times the permitted norm and a different equilibrium.
    expected_acceleration = np.full(3, limit/np.sqrt(3))
    state, thrust_acceleration = equilibrium_state(expected_acceleration)
    options = dict(target_pos=torch.ones(1, 3), target_vel=torch.ones(1, 3),
                   target_acc=2*torch.ones(1, 3), target_yaw=torch.zeros(1, 1))
    bounded = control.compute(state, **options, translational_acceleration_limit=limit)
    wrench = rotor_wrench(control, bounded)
    torch.testing.assert_close(wrench[:, :3], torch.zeros(1, 3), atol=2e-4, rtol=0)
    torch.testing.assert_close(wrench[0, 3], control.mass*thrust_acceleration, atol=2e-5, rtol=1e-6)
    unbounded = control.compute(state, **options)
    assert torch.linalg.vector_norm(rotor_wrench(control, unbounded)[:, :3]) > 1.


def test_vertical_request_preserves_gravity_compensation():
    control = controller()
    state, _ = equilibrium_state([0., 0., 0.])
    upward = torch.tensor([[0., 0., 1.]])
    commands = control.compute(state, target_pos=upward, target_vel=upward,
        target_acc=2*upward, target_yaw=torch.zeros(1, 1), translational_acceleration_limit=.8)
    wrench = rotor_wrench(control, commands)
    torch.testing.assert_close(wrench[:, :3], torch.zeros(1, 3), atol=2e-4, rtol=0)
    torch.testing.assert_close(wrench[0, 3], control.mass*(9.81+.8), atol=2e-5, rtol=1e-6)


def test_hover_and_none_retain_existing_rotor_commands():
    control = controller()
    state, _ = equilibrium_state([0., 0., 0.])
    hover = control.compute(state, target_yaw=torch.zeros(1, 1))
    none = control.compute(state, target_yaw=torch.zeros(1, 1), translational_acceleration_limit=None)
    tiny = control.compute(state, target_yaw=torch.zeros(1, 1), translational_acceleration_limit=.001)
    torch.testing.assert_close(none, hover, atol=0, rtol=0)
    torch.testing.assert_close(tiny, hover, atol=0, rtol=0)
    torch.testing.assert_close(rotor_wrench(control, tiny)[0, 3], control.mass*9.81,
                               atol=2e-5, rtol=1e-6)


def test_request_below_budget_keeps_its_physical_equilibrium():
    control = controller()
    state, thrust_acceleration = equilibrium_state([.1, 0., 0.])
    options = dict(target_acc=torch.tensor([[.1, 0., 0.]]), target_yaw=torch.zeros(1, 1))
    original = control.compute(state, **options)
    bounded = control.compute(state, **options, translational_acceleration_limit=.8)
    torch.testing.assert_close(bounded, original, atol=0, rtol=0)
    wrench = rotor_wrench(control, bounded)
    torch.testing.assert_close(wrench[:, :3], torch.zeros(1, 3), atol=2e-4, rtol=0)
    torch.testing.assert_close(wrench[0, 3], control.mass*thrust_acceleration, atol=2e-5, rtol=1e-6)


@pytest.mark.parametrize('limit', [0., -.1, float('nan'), float('inf'), True])
def test_invalid_acceleration_budgets_are_rejected(limit):
    control = controller()
    state, _ = equilibrium_state([0., 0., 0.])
    with pytest.raises(ValueError, match='finite positive scalar'):
        control.compute(state, translational_acceleration_limit=limit)
