"""GUI presentation for forest tasks; control and sensing keep their own clocks."""
from pathlib import Path
import asyncio
import json
import time

import numpy as np

from .viewer_runtime import RenderCadence


class ForestViewer:
    def __init__(self, env, config, output):
        self.env, self.config, self.output = env, config, Path(output)
        for name in ('follow_drone', 'follow_heading', 'keep_open', 'capture_frames'):
            value = config.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f'viewer.{name} must be true or false; got {value!r}')
        self.mission = None
        self.captures, self._capture_tasks = [], []
        self._next_capture = 0.
        self._window = self._status_label = self._target_label = None
        self.cadence = RenderCadence(config.get('render_hz', 15.), on_render=self._before_render)
        env._camera_updates_with_render = True
        env.enable_render(self.cadence)
        # These labels describe actual task progress and the reason for holding.
        import omni.ui as ui
        self._window = ui.Window('Forest inspection', width=390, height=115)
        with self._window.frame:
            with ui.VStack(spacing=5):
                self._status_label = ui.Label('Starting inspection')
                self._target_label = ui.Label('Waiting for observations')

    def _before_render(self, substep):
        if self.config.get('follow_drone', False):
            self.env.update_camera()
        if self.mission is not None:
            mission = self.mission
            self._status_label.text = f'{mission.status} | {len(mission.history)} inspection legs completed'
            target = np.asarray(mission.current_goal.target_world)
            position = self.env.state[0, 0, :3].detach().cpu().numpy()
            self._target_label.text = f'{mission.current_goal.goal_id} | {np.linalg.norm(target-position):.2f} m remaining'
        if self.config.get('capture_frames', False) and self.env.elapsed >= self._next_capture:
            self._capture(f'flight_{len(self.captures):03d}')
            self._next_capture = float(self.env.elapsed)+30.

    def _capture(self, name):
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        viewport = get_active_viewport()
        if viewport is None:
            return
        folder = self.output/'viewport'
        folder.mkdir(exist_ok=True)
        path = folder/(name+'.png')
        state = self.env.state[0, 0, :13].detach().cpu().numpy()
        capture = capture_viewport_to_file(viewport, str(path))
        future = asyncio.ensure_future(capture.wait_for_result())
        self._capture_tasks.append(future)
        self.captures.append(dict(file=str(path), simulated_time=float(self.env.elapsed),
            actual_position=state[:3].tolist(), actual_quaternion_wxyz=state[3:7].tolist(),
            goal_id=(None if self.mission is None else self.mission.current_goal.goal_id)))

    def save(self):
        capture_status = []
        for row, future in zip(self.captures, self._capture_tasks):
            error = None
            if future.done() and not future.cancelled():
                exception = future.exception()
                error = None if exception is None else str(exception)
            capture_status.append(dict(row, complete=future.done() and not future.cancelled(), error=error,
                file_exists=Path(row['file']).exists()))
        value = dict(render=self.cadence.stats(), captures=capture_status,
            physical_dt_seconds=float(self.env.dt), follow_drone=self.config.get('follow_drone', False),
            follow_heading=self.config.get('follow_heading', False),
            scope='actual Isaac viewport; rendering only, safety ages and physics step unchanged')
        (self.output/'viewer.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

    def hold_result(self, app, result):
        if not self.config.get('keep_open', False) or not app.is_running():
            return
        before = float(self.env.sim.current_time)
        self.env.sim.pause()
        self._status_label.text = 'INSPECTION COMPLETED' if result.get('passed') else 'INSPECTION STOPPED'
        self._target_label.text = f'{result.get("reason", "unknown")} | Close the simulator window to exit'
        print(f'FOREST_VIEWER_RESULT reason={result.get("reason")} passed={result.get("passed")}; window remains open', flush=True)
        if self.config.get('capture_frames', False):
            self._capture('terminal')
        began = time.monotonic()
        duration = self.config.get('keep_open_seconds')
        try:
            while app.is_running() and (duration is None or time.monotonic()-began < float(duration)):
                # SimulationContext.render temporarily disables physics even if
                # Play is clicked. No env.step, scan or task update occurs here.
                self.env.sim.pause()
                self.env.sim.render()
                time.sleep(1./15.)
        finally:
            after = float(self.env.sim.current_time)
            (self.output/'terminal_viewer.json').write_text(json.dumps(dict(
                simulated_time_before=before, simulated_time_after=after,
                physics_time_advanced=after-before, wall_seconds=time.monotonic()-began,
                task_result_mutated=False), indent=2)+'\n')
            self.save()
