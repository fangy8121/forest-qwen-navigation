from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from omni_drones.navigation.forest.local import LocalPlanner


def flight():
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2]/'cfg/task/ForestSurvey.yaml')
    return SimpleNamespace(**dict(cfg.flight))


def test_forward_extent_sets_the_search_window():
    default, longer = LocalPlanner(flight()), LocalPlanner(flight(), grid_forward=4.4)
    assert np.isclose(default.centers[:, 0].max(), 3.2) and np.isclose(longer.centers[:, 0].max(), 4.4)
    assert longer.shape[1:] == default.shape[1:]


@pytest.mark.parametrize('value', [1.2, 6., float('nan')])
def test_forward_extent_stays_inside_the_observed_snapshot(value):
    with pytest.raises(ValueError):
        LocalPlanner(flight(), grid_forward=value)
