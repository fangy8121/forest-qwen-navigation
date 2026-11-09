"""单环境目标条件 PPO 任务；复用 ForestSurvey 的真实旋翼、Lee 与 MID-360。"""
import time
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from .forest_survey import ForestSurvey
from omni_drones.navigation.forest.guidance import Guidance, GuidanceTracker
from omni_drones.navigation.forest.geometry import local_to_world, yaw_from_quat
from omni_drones.navigation.forest.local import LocalMap, VelocityTrajectory, certificate, BODY_NUMERIC_TOLERANCE
from omni_drones.navigation.forest.local_policy import LocalObservation, bound_velocity, local_reward, supervise_velocity
from omni_drones.navigation.forest.planning import PathSafety
from omni_drones.navigation.forest.trajectory import QuinticPath


class ForestLocal(ForestSurvey):
    def __init__(self, cfg, headless):
        self.encoder = LocalObservation(cfg.task.grid_resolution)
        self.rng = np.random.default_rng(cfg.seed)
        super().__init__(cfg, headless)
        self.action_steps = round(1/cfg.local.hz/self.dt)
        self.base_yaw = self.scene_info['initial_yaw']
        self.episode_result = None

    def _set_specs(self):
        super()._set_specs()
        self.observation_spec = Composite({'agents': {'observation': Unbounded((1, self.encoder.size))}}).expand(1).to(self.device)
        self.action_spec = Composite({'agents': {'action': Unbounded((1,3))}}).expand(1).to(self.device)

    def _physics(self, velocity, acceleration):
        action = torch.as_tensor(np.r_[velocity, acceleration, self.scene_info['initial_yaw']],
                                 dtype=torch.float32, device=self.device).reshape(1,1,7)
        super()._step(TensorDict({'agents': {'action': action}}, [1]))
        state = self.state[0,0,:13].detach().cpu().numpy()
        self.trace.append(np.r_[self.elapsed, velocity, acceleration, self.scene_info['initial_yaw'], state])

    def _observe(self):
        state = self.state[0,0,:13].detach().cpu().numpy().copy()
        yaw = yaw_from_quat(state[3:7])
        self.tracker.update(state[:3], yaw, self.elapsed)
        self.memory.add(self.scan())
        points, evidence = self.memory.snapshot(self.elapsed, state[:3])
        g = self.tracker.guidance
        self.safety = PathSafety(points, evidence, g.corridor, self.cfg.task.flight, self.cfg.local.hz, g.height_bounds)
        self.safety.frame_position, self.safety.frame_yaw = state[:3], yaw
        observation = self.encoder.encode(self.safety, state, self.tracker, self.acceleration,
            self.executed, self.shield, self.local_reason, max(0., 1-self.age/self.cfg.task.episode_seconds))
        tensor = torch.as_tensor(observation, device=self.device).reshape(1,1,-1)
        return TensorDict({'agents': {'observation': tensor}}, [1], device=self.device)

    def _advance(self, path, offset=0.):
        for index in range(self.action_steps):
            _, velocity, acceleration = path.sample(offset+index*self.dt)
            self._physics(velocity, acceleration)
            if self.reason:
                break
        _, self.velocity, self.acceleration = path.sample(offset+(index+1)*self.dt)

    def _reset(self, tensordict=None, **kwargs):
        start, end = np.asarray(self.scene_info['start']), np.asarray(self.scene_info['goal'])
        self.scene_info['initial_yaw'] = self.base_yaw+self.rng.uniform(-self.cfg.task.yaw_noise, self.cfg.task.yaw_noise)
        super()._reset(tensordict, **kwargs)
        if self.cfg.task.scene.source == 'empty':
            goal = start+[self.rng.uniform(.6, 1.)*self.cfg.task.goal_span,
                          self.rng.uniform(-1., 1.), self.rng.uniform(0., .5)]
        else:
            goal = start+(end-start)*min(1., self.cfg.task.goal_span/np.linalg.norm(end-start))
        b = self.scene_info['validation']['route_bounds']
        corridor = np.array([[b[0],b[2]], [b[1],b[2]], [b[1],b[3]], [b[0],b[3]]])
        guidance = Guidance('local_task', 1, 'segment', np.array([start, goal]),
            speed_limit=self.cfg.local.speed_limit, corridor=corridor,
            height_bounds=(float(min(start[2],end[2])-.8), float(max(start[2],end[2])+.8)))
        self.goal, self.tracker = goal, GuidanceTracker(guidance, self.cfg.local.lookahead)
        self.memory = LocalMap(guidance.points, self.cfg.history_seconds)
        self.velocity, self.acceleration, self.executed = np.zeros(3), np.zeros(3), np.zeros(3)
        self.shield, self.local_reason, self.age, self.return_sum = False, '', 0., 0.
        self.trace, self.action_log, self.episode_result = [], [], None
        for index in range(round(self.cfg.task.flight.settle_seconds/self.dt)):
            self._physics(np.zeros(3), np.zeros(3))
            if index % self.scan_steps == 0:
                self.memory.add(self.scan())
        # 沿用经典基线的已认证观察动作；策略采样前完成制动，消除起步盲区。
        self._observe()
        self.initial_observation = []
        target = [0.,0.,min(.15, self.cfg.local.speed_limit)]
        for _ in range(round(self.cfg.task.observation_seconds*self.cfg.local.hz)):
            position = self.state[0,0,:3].detach().cpu().numpy()
            path = VelocityTrajectory(position, self.velocity, self.acceleration, target, self.cfg.task.flight)
            rejection, _ = certificate(self.safety, path, position, 2/self.cfg.local.hz)
            self.initial_observation.append(dict(target=target, rejection=rejection))
            if rejection:
                break
            self._advance(path)
            self._observe()
        position = self.state[0,0,:3].detach().cpu().numpy()
        stop = QuinticPath.brake(position, self.velocity, self.acceleration,
                                self.cfg.task.flight.max_acceleration, self.cfg.task.flight.max_jerk)
        envelope = stop.remaining(0., position)
        if not (self.safety.clear(envelope) and self.safety.known(envelope, position, body_padding=BODY_NUMERIC_TOLERANCE)):
            raise RuntimeError('initial observation has no certified brake')
        for index in range(int(np.ceil(stop.duration*self.cfg.local.hz))+3):
            self._advance(stop, index/self.cfg.local.hz)
            observation = self._observe()
        self.tracker.last_progress = self.elapsed
        self.started = self.elapsed
        self.before = float(np.linalg.norm(self.state[0,0,:3].detach().cpu().numpy()-self.goal))
        return observation

    def _step(self, tensordict):
        clock = time.perf_counter()
        state = self.state[0,0,:13].detach().cpu().numpy().copy()
        action = tensordict['agents','action'][0,0].detach().cpu().numpy()
        local = bound_velocity(action, min(self.cfg.local.speed_limit, self.cfg.task.flight.max_speed))
        raw = local_to_world([local], np.zeros(3), yaw_from_quat(state[3:7]))[0]
        previous = self.executed.copy()
        path, self.executed, first_reason, certified, envelope = supervise_velocity(
            self.safety, state[:3], self.velocity, self.acceleration, raw, 2/self.cfg.local.hz)
        self.shield = bool(np.any(self.executed != raw) or not certified)
        self.local_reason = first_reason if self.shield else ''
        self._advance(path)
        self.age = self.elapsed-self.started
        observation = self._observe()
        after = float(np.linalg.norm(self.state[0,0,:3].detach().cpu().numpy()-self.goal))
        speed = float(np.linalg.norm(self.state[0,0,7:10].detach().cpu().numpy()))
        success = after < self.cfg.task.goal_radius and speed < self.cfg.task.flight.settle_speed
        failed = bool(self.reason) or not certified
        stalled = self.elapsed-self.tracker.last_progress > self.cfg.local.no_progress_seconds
        deadline = self.age >= self.cfg.task.episode_seconds
        done = success or failed or stalled or deadline
        reward = local_reward(self.before, after, self.tracker.error, raw, self.executed, previous,
                              success, (failed or stalled or deadline) and not success, self.cfg.task.reward)
        self.before, self.return_sum = after, self.return_sum+float(reward)
        self.action_log.append(dict(timestamp=self.elapsed, policy_action=action.tolist(), raw_velocity=raw.tolist(),
            executed_velocity=self.executed.tolist(), shield=self.shield, reason=self.local_reason,
            certified=certified, reward=float(reward), envelope=envelope.tolist(),
            processing_seconds=time.perf_counter()-clock))
        if done:
            reason = 'goal_reached' if success else self.reason or ('uncertified_stop' if not certified else
                     'no_progress' if stalled else 'episode_deadline')
            self.episode_result = dict(reason=reason, success=success, return_sum=self.return_sum,
                goal_distance=after, progress=self.tracker.progress, seconds=self.age,
                actions=len(self.action_log), shield_actions=sum(a['shield'] for a in self.action_log),
                certified_actions=sum(a['certified'] for a in self.action_log),
                start=self.tracker.points[0].tolist(), goal=self.goal.tolist())
        # 任务有显式完成期限；截止/停滞也是任务终止，不能跨 reset 传播 GAE。
        observation.set(('agents','reward'), torch.tensor([[[reward]]], device=self.device, dtype=torch.float32))
        observation.set('done', torch.tensor([[done]], device=self.device))
        observation.set('terminated', torch.tensor([[done]], device=self.device))
        observation.set('truncated', torch.tensor([[False]], device=self.device))
        return observation
