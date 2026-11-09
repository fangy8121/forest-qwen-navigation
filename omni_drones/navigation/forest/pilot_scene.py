"""Fixed, simplified physical diagnostics; not a forest/VLN benchmark."""
from .subgoal import Subgoal


def cases(timeout=40.):
    from forest_generator.executor_pilot import diagnostic_tasks
    return [Subgoal(**task, timeout_seconds=timeout) for task in diagnostic_tasks()]


def make_scene():
    from forest_generator.executor_pilot import diagnostic_scene
    return diagnostic_scene()
