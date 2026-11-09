"""验证 XYZ 异步执行与 Lee 跟踪；协议桩不计入 Qwen 模型评测。"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from hydra import compose, initialize_config_dir
from omni_drones.navigation.forest import qwen
from omni_drones.navigation.forest.flight import run_flight
from omni_drones.navigation.forest.geometry import PROTOCOL_VERSION
from omni_drones.navigation.forest.observation import representation_spec
from omni_drones.navigation.forest.teacher import teacher_path


class DelayedProtocolStub:
    def __init__(self, *args):
        self.metadata = dict(test_only=True, model='teacher-backed XYZ protocol stub, NOT Qwen',
                             protocol_version=PROTOCOL_VERSION, representation=representation_spec())
        self.calls = 0

    def generate(self, image, prompt):
        start = time.perf_counter()
        time.sleep(.3)
        self.calls += 1
        answer = dict(status='HOLD', points=[])
        if self.calls == 1:
            with np.load(Path(image).parent/'observation.npz') as data:
                obs = dict(data)
            flight = SimpleNamespace(**json.loads(str(obs['flight_json'])))
            answer, _ = teacher_path(obs['points'], obs['goal'], flight,
                obs['velocity'], obs['acceleration'], obs['corridor'], obs['evidence'])
        return json.dumps(answer), time.perf_counter()-start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-id', default='hard/vertical/000000_offset')
    args = parser.parse_args()
    qwen.QwenPlanner = DelayedProtocolStub
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'cfg'), version_base=None):
        cfg = compose(config_name='forest', overrides=[
            'qwen_python=null', 'headless=true', f'task.scene.map_id={args.map_id}',
            'planning.hz=0.2', 'planning.max_snapshot_age=6.0', 'max_seconds=15',
            'output=outputs/debug/closed_loop_xyz'])
    run_flight(cfg, closed_loop=True)
