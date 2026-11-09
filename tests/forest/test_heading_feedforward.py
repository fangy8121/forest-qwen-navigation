from pathlib import Path
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from omni_drones.controllers.lee_position_controller import LeePositionController
from omni_drones.navigation.forest.execution import ContinuousReferencePlayer, course_reference
from forest_generator.multirow_routes import continuous_reference, reference_phases


def test_yaw_feedforward_does_not_brake_correct_tracking():
    params = yaml.safe_load(Path('omni_drones/robots/assets/usd/iris.yaml').read_text())
    controller = LeePositionController(9.81, params)
    state = torch.zeros(1, 13); state[:, 3] = 1.
    hover = controller.compute(state, target_yaw=torch.zeros(1, 1))
    for rate in [-.6, .6]:
        state[:, 12] = rate
        following = controller.compute(state, target_yaw=torch.zeros(1, 1),
            target_yaw_rate=torch.tensor([[rate]]), target_yaw_acceleration=torch.zeros(1, 1))
        braking = controller.compute(state, target_yaw=torch.zeros(1, 1))
        torch.testing.assert_close(following, hover)
        assert not torch.allclose(braking, hover)
    state[:, 12] = 0
    zero = controller.compute(state, target_yaw=torch.zeros(1, 1),
        target_yaw_rate=torch.zeros(1, 1), target_yaw_acceleration=torch.zeros(1, 1))
    torch.testing.assert_close(zero, hover, atol=0, rtol=0)


def test_analytic_course_derivatives_match_curve_and_player_clock():
    flight = OmegaConf.load('cfg/task/ForestSurvey.yaml').flight
    lanes = [dict(reference=[[0., 0., 2.5], [20., 0., 2.5]]),
             dict(reference=[[20., 5., 2.5], [0., 5., 2.5]])]
    path, intervals, _ = continuous_reference(lanes, flight, lambda p: True)
    phases = reference_phases(path, intervals)
    turn = next(p for p in phases if p['phase'] == 'TURN')
    ts = np.linspace(turn['start_s']+.2, turn['end_s']-.2, 3001)
    values = np.array([course_reference(path, t)[3:] for t in ts])
    yaw = np.unwrap(values[:, 0])
    np.testing.assert_allclose(np.gradient(yaw, ts)[2:-2], values[2:-2, 1], atol=2e-4)
    np.testing.assert_allclose(np.gradient(values[:, 1], ts)[2:-2], values[2:-2, 2], atol=2e-3)
    player = ContinuousReferencePlayer(path, phases, 0, settle_seconds=0)
    player.time = turn['start_s']
    assert player.metadata()['execution_state'] == 'U_TURN'
    assert not player.metadata()['default_navigation_training']
    state = np.r_[path.sample(player.time)[0], 1., 0., 0., 0., np.zeros(6)]
    previous = player.time
    player.advance(.01, state, 30)
    assert player.time > previous  # No forced stop at the U-turn boundary.
