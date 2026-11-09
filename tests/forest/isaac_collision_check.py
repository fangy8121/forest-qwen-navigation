"""显式运行的 PhysX 碰撞包络检查，不纳入普通 CPU pytest。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from omni_drones import init_simulation_app


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'cfg'),version_base=None):
        cfg=compose(config_name='forest',overrides=['task.sim.device=cpu','task.sim.use_gpu=false',
            'task.sim.use_gpu_pipeline=false','task.scene.length=8'])
    app=init_simulation_app(cfg)
    from pxr import UsdGeom, UsdPhysics
    from omni_drones.envs.isaac_env import IsaacEnv
    env=IsaacEnv.REGISTRY['ForestSurvey'](cfg,True)
    start=np.array([0.,0.,2.])
    env.scene_info['reference'][0]=start.tolist()
    env.reset()
    action=torch.tensor(np.r_[start,np.zeros(7)],dtype=torch.float32).reshape(1,1,10)
    for _ in range(3): env.step(TensorDict({'agents':{'action':action}},[1]))
    checks=[dict(case='verified corridor',contact=bool(env.reason=='collision'),expected=False)]
    mesh_prim=env.sim.stage.GetPrimAtPath('/World/envs/env_0/forest_mesh')
    envelope=env.sim.stage.GetPrimAtPath('/World/envs/env_0/Iris_0/base_link/survey_envelope')
    radius=float(UsdGeom.Cylinder(envelope).GetRadiusAttr().Get())
    checks.append(dict(case='mesh and envelope collider authored',
                       forest_collision_api=bool(mesh_prim.HasAPI(UsdPhysics.CollisionAPI)),
                       forest_mesh_collision_api=bool(mesh_prim.HasAPI(UsdPhysics.MeshCollisionAPI)),
                       envelope_collision_api=bool(envelope.HasAPI(UsdPhysics.CollisionAPI)),
                       envelope_radius=radius, expected=True))
    # 只在独立测试 reset 时放置轻微重叠，验证新增包络确实产生树干接触力。
    tree=min(env.scene_info['trees'],key=lambda t:np.linalg.norm(np.asarray(t['position'])-[4,0]))
    probe=np.r_[np.asarray(tree['position'])+[tree['radius']+.2,0],2.]
    env.scene_info['reference'][0]=probe.tolist()
    env.reset()
    action=torch.tensor(np.r_[probe,np.zeros(6),env.scene_info['initial_yaw']],dtype=torch.float32).reshape(1,1,10)
    contact=False
    for _ in range(10):
        env.step(TensorDict({'agents':{'action':action}},[1]))
        contact |= env.reason=='collision'
    checks.append(dict(case='tree contact response',contact=contact,expected=True,
                       probe_position=probe.tolist(),actual_position=env.state[0,0,:3].tolist()))
    result=dict(dimensions=env.dimensions,physics_mass=env.physics_mass,checks=checks,
                passed=bool(not checks[0]['contact'] and checks[1]['forest_collision_api'] and
                        checks[1]['forest_mesh_collision_api'] and
                        checks[1]['envelope_collision_api'] and np.isclose(radius,.5) and
                        np.isclose(env.dimensions['height'],.11,atol=.002) and contact))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert result['passed'], 'collision geometry audit failed'
    env.close();app.close()


if __name__=='__main__': main()
