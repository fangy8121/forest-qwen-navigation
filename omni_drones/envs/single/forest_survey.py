"""单机树行巡检；控制 action 为参考 p/v/a/yaw，而非直接设置位姿。"""
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema, Vt
import isaacsim.core.utils.stage as stage_utils
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.drone import MultirotorBase
from omni_drones.navigation.forest.scene import create_scene
from omni_drones.navigation.forest.geometry import RADIUS, MARGIN, clearance
from omni_drones.navigation.forest.lidar import Mid360


class ForestSurvey(IsaacEnv):
    def __init__(self, cfg, headless):
        if cfg.env.num_envs != 1 or cfg.sim.substeps != 1:
            raise ValueError('ForestSurvey requires one drone and substeps=1')
        if cfg.task.drone_model.name != 'Iris' or cfg.task.drone_model.controller != 'LeePositionController':
            raise ValueError('ForestSurvey uses OmniDrones Iris + LeePositionController')
        self.scene_info, self.mesh_vertices, self.mesh_faces, _ = create_scene(cfg.seed, **dict(cfg.task.scene))
        if cfg.viewer.get('follow_start', False):
            offset = np.asarray(self.scene_info['reference'][0]) - [0., 0., 2.]
            cfg.viewer.eye = (np.asarray(cfg.viewer.eye) + offset).tolist()
            cfg.viewer.lookat = (np.asarray(cfg.viewer.lookat) + offset).tolist()
        self.elapsed = 0.
        self.reason = None
        self.velocity_control = cfg.task.get('control_mode', 'trajectory') == 'velocity'
        super().__init__(cfg, headless)
        self.drone.initialize(track_contact_forces=True)
        self.controller.to(self.device)
        self._verify_mass()
        from isaaclab.utils.warp import convert_to_warp_mesh
        self.ray_mesh = convert_to_warp_mesh(self.mesh_vertices, self.mesh_faces.reshape(-1), str(self.device))
        l = cfg.task.lidar
        self.lidar = Mid360(l.omniperception_root, self.ray_mesh, self.device,
            l.max_range, l.downsample, l.mount_position, l.mount_quaternion,
            downward=dict(l.downward) if l.get('downward') is not None else None)
        if not np.isclose((1/l.hz)/self.dt, round((1/l.hz)/self.dt)):
            raise ValueError('LiDAR period must be a multiple of physics dt')
        self.scan_steps = round(1/l.hz/self.dt)
        self.reference = np.array(self.scene_info['reference'][0]+[0.]*6+[0.])
        self._display_reference_id = None
        self._display_time = -np.inf
        start = np.asarray(self.scene_info['reference'][0])
        self._camera_eye_offset = np.asarray(cfg.viewer.eye)-start
        self._camera_target_offset = np.asarray(cfg.viewer.lookat)-start

    def update_camera(self, *, force=False):
        """Follow actual position, or frame a new task spawn once for free orbit."""
        if self.enable_viewport and (force or self.cfg.viewer.get('follow_drone', False)):
            from isaacsim.core.utils.viewports import set_camera_view
            position = self.state[0,0,:3].detach().cpu().numpy()
            eye, target = self._camera_eye_offset, self._camera_target_offset
            if self.cfg.viewer.get('follow_heading', False):
                from omni_drones.navigation.forest.geometry import yaw_from_quat
                yaw = yaw_from_quat(self.state[0,0,3:7].detach().cpu().numpy())
                c, s = np.cos(yaw), np.sin(yaw)
                rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                eye, target = rotation@eye, rotation@target
            set_camera_view(eye=position+eye, target=position+target)

    def _design_scene(self):
        stage = stage_utils.get_current_stage()
        # RTX 视口需要显式光源，否则物理和扫描正常时画面仍会全黑。
        # 灯光只影响显示，不加入碰撞体或 LiDAR 求交网格。
        sky = UsdLux.DomeLight.Define(stage, '/World/forest_sky_light')
        sky.CreateIntensityAttr(700.)
        sun = UsdLux.DistantLight.Define(stage, '/World/forest_sun_light')
        sun.CreateIntensityAttr(1500.)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35., 25., 15.))
        self.drone, self.controller = MultirotorBase.make('Iris', 'LeePositionController', self.device)
        root = self.drone.spawn(translations=[self.scene_info['reference'][0]])[0]
        self._add_envelope(stage, root)
        # 同一份随机枝干网格同时作为可见几何、PhysX 碰撞体和 Warp 射线场景。
        if 'map_path' in self.scene_info:
            from pathlib import Path
            prim = stage.DefinePrim(f'{self.template_env_ns}/forest', 'Xform')
            prim.GetReferences().AddReference(str(Path(self.scene_info['map_path']) / self.scene_info['usd_file']))
        else:
            mesh = UsdGeom.Mesh.Define(stage, f'{self.template_env_ns}/forest_mesh')
            mesh.CreatePointsAttr(self.mesh_vertices.tolist())
            mesh.CreateFaceVertexCountsAttr([3]*len(self.mesh_faces))
            mesh.CreateFaceVertexIndicesAttr(self.mesh_faces.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDisplayColorAttr([(.38, .43, .29)])
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        self.live_points, self.live_recent, self.live_reference, self.live_actual = self._make_display_prims(stage)
        return []

    def _make_display_prims(self, stage):
        if not self.enable_viewport:
            return (None,)*4
        points = []
        for name, color in [('current_scan', (.1,.9,1.)), ('recent_cloud', (.25,.55,1.))]:
            prim = UsdGeom.Points.Define(stage, f'{self.template_env_ns}/{name}')
            prim.CreateDisplayColorAttr([color]); prim.CreateWidthsAttr([.025])
            prim.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            points.append(prim)
        curves = []
        for name, color in [('reference_path', (.15,.8,.2)), ('actual_path', (1.,.2,.1))]:
            curve = UsdGeom.BasisCurves.Define(stage, f'{self.template_env_ns}/{name}')
            curve.CreateTypeAttr(UsdGeom.Tokens.linear); curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            curve.CreateWidthsAttr([.035]); curve.CreateDisplayColorAttr([color]); curves.append(curve)
            curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        return (*points, *curves)

    def update_display(self, frame, history, reference, actual):
        if not self.enable_viewport:
            return
        if self.elapsed-self._display_time < .2-1e-6:
            return
        self._display_time = self.elapsed
        from omni_drones.navigation.forest.geometry import quat_matrix
        from pxr import Vt
        def world_points(source):
            if source is None:
                return np.empty((0,3))
            valid = source['valid']
            rotation = quat_matrix(source['sensor_quaternion'])
            return source['points_sensor'][valid] @ rotation.T + source['sensor_position']
        current = world_points(frame)
        accumulated = np.concatenate([world_points(f) for f in history.frames]) if history.frames else np.empty((0,3))
        for prim, array in ((self.live_points,current), (self.live_recent,accumulated)):
            # Bound viewport cost only; raw LiDAR frames remain complete.
            array = array[::max(1, int(np.ceil(len(array)/12000)))]
            prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(array, dtype=np.float32)))
        for prim, path in ((self.live_reference,reference), (self.live_actual,actual)):
            if prim == self.live_reference and self._display_reference_id is reference:
                continue
            array = np.asarray(path).reshape(-1,3)
            if len(array) < 2:
                continue
            indices = np.unique(np.r_[np.arange(0, len(array), max(1, int(np.ceil(len(array)/6000)))), len(array)-1])
            array = array[indices]
            prim.GetCurveVertexCountsAttr().Set([len(array)])
            prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(array, dtype=np.float32)))
            if prim == self.live_reference:
                self._display_reference_id = reference

    def _add_envelope(self, stage, root):
        base = stage.GetPrimAtPath(str(root.GetPath())+'/base_link')
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        base_inverse = cache.GetLocalToWorldTransform(base).GetInverse()
        zmin, zmax = np.inf, -np.inf
        self.original_mass = {}
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                # 必须在 PhysX 首次解析场景前启用报告，否则运行中新增 API 可能没有接触数据。
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
            if p.HasAPI(UsdPhysics.MassAPI):
                api = UsdPhysics.MassAPI(p)
                mass = api.GetMassAttr().Get()
                inertia = api.GetDiagonalInertiaAttr().Get()
                if mass <= 0 or min(inertia) <= 0:
                    raise ValueError('asset must have explicit positive mass and inertia')
                self.original_mass[p.GetName()] = dict(mass=mass, diagonal_inertia=list(inertia))
            if p.HasAPI(UsdPhysics.CollisionAPI):
                if p.IsA(UsdGeom.Cube):
                    half = np.full(3, UsdGeom.Cube(p).GetSizeAttr().Get()/2)
                elif p.IsA(UsdGeom.Cylinder):
                    shape = UsdGeom.Cylinder(p)
                    half = np.full(3, shape.GetRadiusAttr().Get())
                    half['XYZ'.index(shape.GetAxisAttr().Get())] = shape.GetHeightAttr().Get()/2
                else:
                    raise ValueError(f'unsupported collider shape for size audit: {p.GetPath()} ({p.GetTypeName()})')
                # 按 PhysX 实际使用的 Cube/Cylinder 尺寸，变换到机体坐标审计垂直包络。
                transform = cache.GetLocalToWorldTransform(p)*base_inverse
                for signs in [(x,y,z) for x in [-1,1] for y in [-1,1] for z in [-1,1]]:
                    corner = transform.Transform(Gf.Vec3d(*(half*np.array(signs)).tolist()))
                    zmin, zmax = min(zmin, corner[2]), max(zmax, corner[2])
        if not np.isfinite([zmin,zmax]).all() or zmax <= zmin or zmax-zmin > 2.0:
            raise ValueError(f'cannot measure original collider vertical extent: z_min={zmin}, z_max={zmax}')
        # 只新增碰撞包络，不缩放模型、不改变动力学。显式质量/惯量阻止自动重算。
        api = UsdPhysics.MassAPI(base)
        api.CreateMassAttr(self.original_mass['base_link']['mass'])
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*self.original_mass['base_link']['diagonal_inertia']))
        envelope = UsdGeom.Cylinder.Define(stage, str(base.GetPath())+'/survey_envelope')
        envelope.CreateRadiusAttr(RADIUS); envelope.CreateHeightAttr(zmax-zmin)
        envelope.CreateAxisAttr('Z')
        envelope.AddTranslateOp().Set(Gf.Vec3d(0,0,(zmin+zmax)/2))
        envelope.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(envelope.GetPrim())
        self.dimensions = dict(horizontal_radius=RADIUS, safety_margin=MARGIN,
            inflation=RADIUS+MARGIN, z_min=zmin, z_max=zmax, height=zmax-zmin,
            interpretation='collision envelope only; original dynamics retained')

    def _verify_mass(self):
        self.physics_mass = {}
        for label, view in [('base_link', self.drone.base_link), ('rotors', self.drone.rotors_view)]:
            masses = view.get_masses().detach().cpu().numpy().reshape(-1)
            matrices = view.get_inertias().detach().cpu().numpy().reshape(-1,3,3)
            for i, (m, inertia) in enumerate(zip(masses, matrices)):
                name = label if label == 'base_link' else f'rotor_{i}'
                expected = self.original_mass[name]
                if not np.isclose(m, expected['mass'], rtol=1e-5) or not np.allclose(inertia, np.diag(expected['diagonal_inertia']), rtol=1e-4, atol=1e-8):
                    raise RuntimeError(f'PhysX mass/inertia changed after envelope: {name}, {m}, {inertia}')
                self.physics_mass[name] = dict(mass=float(m), inertia=inertia.tolist())

    def _set_specs(self):
        self.observation_spec = Composite({'agents': {'observation': self.drone.state_spec.unsqueeze(0)}}).expand(1).to(self.device)
        self.action_spec = Composite({'agents': {'action': Unbounded((1,7 if self.velocity_control else 10))}}).expand(1).to(self.device)
        self.reward_spec = Composite({'agents': {'reward': Unbounded((1,1))}}).expand(1).to(self.device)
        self.agent_spec['drone'] = AgentSpec('drone', 1, observation_key=('agents','observation'), action_key=('agents','action'), reward_key=('agents','reward'))

    def _reset_idx(self, env_ids):
        self.drone._reset_idx(env_ids)
        pos = torch.tensor([self.scene_info['reference'][0]], device=self.device)
        yaw = self.scene_info['initial_yaw']
        q = torch.tensor([[np.cos(yaw/2),0,0,np.sin(yaw/2)]], dtype=torch.float32, device=self.device)
        # 仅 reset 允许设置位姿，后续所有运动由旋翼力产生。
        self.drone.set_world_poses(pos+self.envs_positions[env_ids], q, env_ids)
        self.drone.set_velocities(torch.zeros(1,1,6,device=self.device), env_ids)
        self.elapsed, self.reason = 0., None
        self.lidar.reset()
        self.reference = np.r_[self.scene_info['reference'][0], np.zeros(7)]
        self.heading_feedforward = None

    def set_heading_feedforward(self, rate, acceleration):
        limits = self.cfg.task.flight
        if (not np.isfinite([rate, acceleration]).all() or abs(rate) > limits.max_yaw_rate*1.01
                or abs(acceleration) > limits.max_yaw_acceleration*1.01):
            raise ValueError('Heading feedforward exceeds the map motion limits')
        self.heading_feedforward = (float(rate), float(acceleration))

    def _pre_sim_step(self, tensordict):
        self._physics_time_before = self.sim.current_time
        reference = tensordict['agents','action']
        state = self.drone.get_state(env_frame=False)[...,:13]
        options = {}
        if self.heading_feedforward is not None:
            options = dict(zip(('target_yaw_rate', 'target_yaw_acceleration'),
                [torch.full_like(reference[...,-1:], value) for value in self.heading_feedforward]))
        tracking_limit = self.cfg.task.get('tracking_acceleration_limit')
        if tracking_limit is not None:
            options['translational_acceleration_limit'] = tracking_limit
        if self.velocity_control:
            self.reference = np.r_[state[0,0,:3].detach().cpu().numpy(), reference[0,0].detach().cpu().numpy()]
            motors = self.controller.compute(state, target_vel=reference[...,:3],
                target_acc=reference[...,3:6], target_yaw=reference[...,6:7], **options)
        else:
            self.reference = reference[0,0].detach().cpu().numpy()
            motors = self.controller.compute(state, reference[...,:3], reference[...,3:6], reference[...,6:9], reference[...,9:10], **options)
        self.motor_commands = motors.detach().cpu().numpy()
        self.drone.apply_action(motors)

    def _post_sim_step(self, tensordict):
        # A GUI pause can occur inside a rendered step. Never advance a
        # reference/scan clock when PhysX did not advance.
        self.elapsed += max(0., self.sim.current_time-self._physics_time_before)

    def _compute_state_and_obs(self):
        self.state = self.drone.get_state(env_frame=False)
        return TensorDict({'agents': {'observation': self.state}}, self.batch_size)

    def _compute_reward_and_done(self):
        s = self.state[0,0].detach().cpu().numpy()
        error = np.linalg.norm(s[:3]-self.reference[:3])
        contact = max(self.drone.base_link.get_net_contact_forces().norm().item(), self.drone.rotors_view.get_net_contact_forces().norm().item())
        if not np.isfinite(s).all():
            self.reason = 'nonfinite_state'
        elif contact > 1e-5:
            self.reason = 'collision'
        elif not self.velocity_control and error > self.cfg.task.flight.tracking_limit:
            self.reason = 'tracking_failure'
        elif self.progress_buf.item() >= self.max_episode_length:
            self.reason = 'timeout'
        done = torch.tensor([[self.reason is not None]], device=self.device)
        truncated = torch.tensor([[self.reason == 'timeout']], device=self.device)
        return TensorDict({'agents': {'reward': torch.zeros(1,1,1,device=self.device)},
            'done': done, 'terminated': done & ~truncated, 'truncated': truncated}, self.batch_size)

    def scan(self):
        state = self.state[0,0].detach().cpu().numpy()
        frame = self.lidar.scan(self.elapsed, state[:3], state[3:7])
        frame['velocity'] = state[7:13].copy()
        return frame
