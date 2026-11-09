"""GUI 手动验收脚本：检查网格森林、机体和滚动 MID-360 点云均在真实视口显示。"""
import argparse
import asyncio
import json
import time
from pathlib import Path
import torch
from tensordict import TensorDict
from hydra import compose, initialize_config_dir
from omni_drones import init_simulation_app
from omni_drones.navigation.forest.observation import History

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
args = p.parse_args()
folder = Path(args.output).resolve()
folder.mkdir(parents=True, exist_ok=False)
with initialize_config_dir(version_base=None, config_dir=str(Path('cfg').resolve())):
    cfg = compose(config_name='forest', overrides=['headless=false', 'seed=101', 'task.scene.length=8'])
app = init_simulation_app(cfg)
from omni_drones.envs.isaac_env import IsaacEnv
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from pxr import UsdGeom, UsdLux

env = IsaacEnv.REGISTRY['ForestSurvey'](cfg, False)
env.reset()
history = History(3.)
reference = torch.tensor([0.,0.,2.,0.,0.,0.,0.,0.,0.,0.], device=env.device).reshape(1,1,10)
td = TensorDict({'agents': {'action': reference}}, [1])
actual_path = []
for step in range(310):
    env.step(td)
    if step % env.scan_steps == 0:
        frame = env.scan()
        history.add(frame)
        state = env.state[0,0].detach().cpu().numpy()
        actual_path.append(state[:3].copy())
        env.update_display(frame, history, env.scene_info['reference'], actual_path)
viewport = get_active_viewport()
assert viewport is not None
capture = capture_viewport_to_file(viewport, str(folder/'viewport.png'))
future = asyncio.ensure_future(capture.wait_for_result())
deadline = time.perf_counter()+30
while not future.done() and time.perf_counter() < deadline:
    env.step(td)
assert future.done(), 'viewport capture timed out'
future.result()
stage = env.sim.stage
info = dict(camera=str(viewport.camera_path), resolution=list(viewport.resolution),
    lights=[str(x.GetPath()) for x in stage.Traverse() if x.HasAPI(UsdLux.LightAPI)],
    forest_meshes=[str(x.GetPath()) for x in stage.Traverse() if x.IsA(UsdGeom.Mesh) and 'forest_mesh' in str(x.GetPath())],
    visible_point_prims=[str(x.GetPath()) for x in stage.Traverse() if x.IsA(UsdGeom.Points) and 'env_0' in str(x.GetPath())],
    scans=len(history.frames), latest_scan_points=int(frame['valid'].sum()),
    actual_position=env.state[0,0,:3].cpu().tolist())
(folder/'scene.json').write_text(json.dumps(info, indent=2)+'\n')
print('VIEWPORT_CAPTURE='+str(folder), flush=True)
env.close(); app.close()
